import scipy.stats as stat


def phi(z):
  ## give the pdf of a standard normal
  return stat.norm(0,1).pdf(z)

def Q(z):
  ## give upper tail pribability of a standard normal
  return 1-stat.norm(0,1).cdf(z)

def upper_quantile(tau):
  ## inverse of Q
  ## return the quantile that corresponds to a upper tail probability
  return stat.norm.ppf(1-tau)

def tau_to_beta(tau, mu, sigma):
  z = upper_quantile(tau)
  beta = Q((z - mu)/sigma)
  return beta

def beta_to_tau(beta, mu, sigma):
  q = upper_quantile(beta)
  z = q * sigma + mu
  tau = Q(z)
  return tau
