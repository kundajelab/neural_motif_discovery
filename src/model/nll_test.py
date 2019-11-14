import numpy as np
import tensorflow as tf
import model.profile_models_mimic as profile_models_mimic

np.random.seed(20191110)
batch_size, num_tasks, prof_len = 100, 5, 1000
prof_shape = (batch_size, num_tasks, prof_len, 2)
true_profs_np = np.random.randint(5, size=prof_shape)
logit_pred_profs_np = np.random.randn(*prof_shape)

# Setup placeholders
true_profs = tf.placeholder(tf.float32, (batch_size, prof_len, 2))
logit_pred_profs = tf.placeholder(tf.float32, (batch_size, prof_len, 2))

# Add the profiles together to get the raw counts
true_counts = tf.reduce_sum(true_profs, axis=)

# Convert logits to log probabilities
log_pred_profs = profile_models.profile_logits_to_log_probs(logit_pred_profs)

# Compute probability of seeing true profile under distribution of log
# predicted probs
nll = -profile_models.multinomial_log_probs(log_pred_profs, true_counts, true_profs)

nll = nll.numpy()
