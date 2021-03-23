import os

def make_job_yaml(template_str, name, tf_name, task_name, fold_num, out_path):
    with open(out_path, "w") as f:
        f.write(template_str.format(
            name=name, tf_name=tf_name, task_name=task_name, fold_num=fold_num
        ))

if __name__ == "__main__":
    base_path = "/users/amtseng/tfmodisco/infra/gcp/jobs/svm/"
    template_path = os.path.join(base_path, "gkmexplain_template.yaml")

    with open(template_path, "r") as f:
        template_str = f.read()

    for tf_name, task_name, fold_num in [
        ("FOXA2", "FOXA2_ENCSR000BNI_HepG2", 7),
        ("FOXA2", "FOXA2_ENCSR066EBK_HepG2", 7),
        ("FOXA2", "FOXA2_ENCSR080XEY_liver", 7),
        ("FOXA2", "FOXA2_ENCSR310NYI_liver", 7),
        ("SPI1", "SPI1_ENCSR000BGQ_GM12878", 8),
        ("SPI1", "SPI1_ENCSR000BGW_K562", 8),
        ("SPI1", "SPI1_ENCSR000BIJ_GM12891", 8),
        ("SPI1", "SPI1_ENCSR000BUW_HL-60", 8),
        ("MAX", "MAX_ENCSR000BLP_K562", 1),
        ("MAX", "MAX_ENCSR000DZF_GM12878", 1),
        ("MAX", "MAX_ENCSR000EDS_HepG2", 1),
        ("MAX", "MAX_ENCSR000EFV_K562", 1),
        ("MAX", "MAX_ENCSR000EUP_H1", 1),
        ("MAX", "MAX_ENCSR521IID_liver", 1),
        ("MAX", "MAX_ENCSR847DIT_liver", 1)
    ]:
        name = task_name.lower().replace("_", "-")
        out_path = os.path.join(base_path, "specs", name + ".yaml")
        make_job_yaml(template_str, name, tf_name, task_name, fold_num, out_path)
