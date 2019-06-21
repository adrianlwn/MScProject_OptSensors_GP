parameters = dict()
parameters["i_start"] = 0
parameters["i_end"] = 20
parameters["crop"] = ((-5, 5), (-5, 5), (0, 20))
parameters["field_name"] = "TracerBackground"


# Covariance Choice : Sample Covariance
 
parameters["cov_method"] = "sample"


# Covariance Choice : Matern Kernel 5/2
""" 
parameters["cov_method"] = "arbitrary"
parameters["kernel"] = matern52
parameters["lengthscale"] = 0.4
parameters["lengthscale"] = 0.4
"""