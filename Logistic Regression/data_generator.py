import pandas as pd
import numpy as np



num_data_points = 100

hours_studied = np.random.uniform(0, 20, num_data_points)
hours_project = np.random.uniform(0, 50, num_data_points)

def calculate_pass_probability(study_hours, project_hours):
    base_pass_probability = 0.02
    exam_coeff = 0.038
    project_coeff= 0.033
    for i in range(num_data_points):
        if study_hours[i] + project_hours[i] < 20:
            base_pass_probability -= 0.055
    return base_pass_probability + (exam_coeff * study_hours + project_coeff * project_hours)


pass_probability = calculate_pass_probability(hours_studied, hours_project)
pass_fail_labels = (np.random.random(num_data_points) < pass_probability).astype(int)

noise_factor = 0
for i in range(num_data_points):
    if np.random.rand() < noise_factor:
        pass_fail_labels[i] = 1 - pass_fail_labels[i]

df = pd.DataFrame({'Hours_Studied': hours_studied, 'Hours_Project': hours_project, 'Pass': pass_fail_labels})

df.to_csv('dataset.csv', index=True)