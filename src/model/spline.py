import keras
import keras.layers as kl
import keras.backend as kb
import numpy as np
import scipy.interpolate

class SplineWeight1D(kl.Layer):
    def __init__(self, num_bases, spline_order=3, use_bias=True, **kwargs):
        """
        Creates a new SplineWeight1D Keras layer. This layer transforms the
        input sequence by weighting the features along the sequence by a spline.
        The spline is constructed as a linear combination of basis (BSpline)
        polynomials (perhaps with a bias). A separate spline (i.e. separate set
        of basis weights) is constructed for each feature in the input sequence.
        Arguments:
            `num_bases`: the number of basis polynomials to use for the spline;
                knots will be chosen such that these basis polynomials are
                evenly spaced through the input sequence
            `spline_order`: order of the spline; the degree of the basis
                polynomials is 1 smaller than the order
            `use_bias`: whether or not to add a bias to the weights, after
                scaling by the spline
        Copied (and slightly modified) from
        https://github.com/gagneurlab/concise/blob/d15262eb1e590008bc96ba31e93bfbdbfa1a9fd4/concise/layers.py#L310
        """
        super(SplineWeight1D, self).__init__(**kwargs)
        
        self.num_bases = num_bases
        self.spline_order = spline_order
        self.use_bias = use_bias
        
    def _get_knots(self, seq_length):
        """
        For a sequence of length `seq_length`, and given the specified number
        of basis polynomials and the spline order, computes the locations of
        the spline knots so that the basis polynomials are equally spaced
        throughout the sequence, and the sum of all basis polynomials will be
        a constant function with value of 1 (in the domain of the sequence).
        Exactly how this calculation works is through dark magic.
        """
        # Set beginning and end indices, with some extra padding
        start, end = 0, seq_length
        start -= seq_length * 0.001
        end += seq_length * 0.001
        
        order = self.spline_order
        degree = order - 1
        num_knots = self.num_bases - degree  # Number of interior knots
        
        knot_diff = (end - start) / (num_knots - 1)
        return np.linspace(
            start=(start - (knot_diff * order)),
            stop=(end + (knot_diff * order)),
            num=(num_knots + (2 * order))
        )
        
    def build(self, input_shape):
        """
        Builds the Keras layer, given the shape of the input tensor.
        """
        super(SplineWeight1D, self).build(input_shape)

        assert len(input_shape) == 3, "Input must be (batch x steps x features)"
        input_length, num_features = input_shape[1], input_shape[2]
        
        # Get the locations of the knots
        knots = self._get_knots(input_length)
        # This will be longer than the number of basis polynomials; just use the
        # first `num_bases` entries
        
        # Evaluate the sequence (i.e. steps) for each basis polynomial
        basis_poly_vals = np.empty((input_length, self.num_bases))
        input_steps = np.arange(input_length)  # x-values for the spline
        for basis_index in range(self.num_bases):
            # Pretend only this basis polynomial has coefficient of 1; all other
            # bases are weighted with 0
            bspline_coeffs = np.zeros(self.num_bases)
            bspline_coeffs[basis_index] = 1
            
            basis_poly_vals[:, basis_index] = scipy.interpolate.splev(
                input_steps, (knots, bspline_coeffs, self.spline_order)
            )  # Evaluate the x-values using only the polynomial at this knot
        self.basis_poly_vals = basis_poly_vals
        # Convert to Keras constant:
        self.basis_poly_vals_tensor = kb.constant(basis_poly_vals)
        
        # Create the weights that weight the basis polynomials in the B-spline;
        # Each feature has a separate set of weights (i.e. each feature gets its
        # own spline)
        self.kernel = self.add_weight(
            shape=(self.num_bases, num_features), initializer="random_normal",
            name="kernel", trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(num_features,), initializer="zeros", name="bias",
                trainable=True
            )

    def call(self, x):
        # x shape: (batch x steps x features)
        
        # Compute the spline by weighting the basis polynomials by the weights
        splines = kb.dot(self.basis_poly_vals_tensor, self.kernel)
        # Note: kb.dot is matrix multiplication
        if self.use_bias:
            splines = kb.bias_add(splines, self.bias)
        
        # Weight the features in x by the spline
        return x * (splines + 1)  # TODO: why add 1?
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "num_bases": self.num_bases,
            "spline_order": self.spline_order,
            "use_bias": self.use_bias
        }
        base_config = super(SplineWeight1D, self).get_config()
        for key, val in config.items():
            base_config[key] = val
        return base_config
    
    def get_basis(self):
        """
        Returns the basis polynomials as an L x B array, where L is the length
        of the input sequence tensor, and B is the number of basis polynomials.
        Each L-vector is a basis polynomial.
        """
        return self.basis_poly_vals
        
    def get_spline(self):
        """
        Returns the multiplicative effect of the layer (given its current
        weights), as an L x F array, where L is the length of the input sequence
        tensor, and F is the number of features (i.e. feature depth). Each
        L-array is the spline (i.e. the linear combination of basis polynomials)
        for that feature. It is the multiplicative effect of the layer on each
        value in a sequence (without any bias added).
        """
        return np.matmul(self.basis_poly_vals, self.get_weights()[0])
