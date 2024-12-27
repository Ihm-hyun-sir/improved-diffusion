from T2IBenchmark import calculate_fid

# for i in range(10):
#     print(f'Class {i}')
fid, _ = calculate_fid('/home/kjh012/MS_Year_Proposal_2024_Winter/improved-diffusion/cifar_noisy_20/cifar_noisy_test',
                    '/home/kjh012/MS_Year_Proposal_2024_Winter/improved-diffusion/sample_original_noisy_symmetric_20/Full')
print(fid)

#sample_original_noisy_symmetric_20
#sample_soft_noisy_symmetric_20
#sample_original_noisy_asymmetric_20
#sample_soft_noisy_asymmetric_20