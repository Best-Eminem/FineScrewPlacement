import numpy as np

num_samples = 5
desired_mean = 2144
desired_std_dev = 186

samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)

actual_mean = np.mean(samples)
# actual_std = np.std(samples,ddof=1)
actual_std = np.std(samples)
print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))

zero_mean_samples = samples - (actual_mean)

zero_mean_mean = np.mean(zero_mean_samples)
zero_mean_std = np.std(zero_mean_samples,ddof=1)
print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
scaled_mean = np.mean(scaled_samples)
scaled_std = np.std(scaled_samples,ddof=1)
print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

final_samples = scaled_samples + desired_mean
final_mean = np.mean(final_samples)
final_std = np.std(final_samples,ddof=1)
print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))

print(final_samples)
print('最大值：',max(final_samples))
print('最小值：',min(final_samples))
print(np.mean(final_samples))
print(np.std(final_samples,ddof=1))
