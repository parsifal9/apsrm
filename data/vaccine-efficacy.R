# Copyright 2022 CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# based on the file "20210511 generate figures_TimsCodeFig_2_3.R"
# available from https://github.com/InfectionAnalytics/COVID19-ProtectiveThreshold
# This is code for the paper
## @article{khouryNeutralizingAntibodyLevels2021,
##   title =	 {Neutralizing Antibody Levels Are Highly Predictive of Immune Protection from Symptomatic
##                   {{SARS}}-{{CoV}}-2 Infection},
##   author =	 {Khoury, David S. and Cromer, Deborah and Reynaldi, Arnold and Schlub, Timothy E. and Wheatley, Adam
##                   K. and Juno, Jennifer A. and Subbarao, Kanta and Kent, Stephen J. and Triccas, James A. and Davenport,
##                   Miles P.},
##   year =	 2021,
##   month =	 jul,
##   journal =	 {Nature Medicine},
##   volume =	 27,
##   number =	 7,
##   pages =	 {1205--1211},
##   publisher =	 {{Nature Publishing Group}},
##   issn =	 {1546-170X},
##   doi =		 {10.1038/s41591-021-01377-8},
##   abstract =	 {Predictive models of immune protection from COVID-19 are urgently needed to identify correlates of
##                   protection to assist in the future deployment of vaccines. To address this, we analyzed the
##                   relationship between in vitro neutralization levels and the observed protection from severe acute
##                   respiratory syndrome coronavirus 2 (SARS-CoV-2) infection using data from seven current vaccines and
##                   from convalescent cohorts. We estimated the neutralization level for 50\% protection against
##                   detectable SARS-CoV-2 infection to be 20.2\% of the mean convalescent level (95\% confidence interval
##                   (CI)\,=\,14.4\textendash 28.4\%). The estimated neutralization level required for 50\% protection from
##                   severe infection was significantly lower (3\% of the mean convalescent level; 95\%
##                   CI\,=\,0.7\textendash 13\%, P\,=\,0.0004). Modeling of the decay of the neutralization titer over the
##                   first 250\,d after immunization predicts that a significant loss in protection from SARS-CoV-2
##                   infection will occur, although protection from severe disease should be largely
##                   retained. Neutralization titers against some SARS-CoV-2 variants of concern are reduced compared with
##                   the vaccine strain, and our model predicts the relationship between neutralization and efficacy
##                   against viral variants. Here, we show that neutralization level is highly predictive of immune
##                   protection, and provide an evidence-based model of SARS-CoV-2 immune protection that will assist in
##                   developing vaccine strategies to control the future trajectory of the pandemic.},
##   copyright =	 {2021 The Author(s), under exclusive licence to Springer Nature America, Inc.},
##   langid =	 {english},
##   keywords =	 {Computational biology and bioinformatics,Vaccines,Viral infection},
##   annotation =	 {Bandiera\_abtest: a Cg\_type: Nature Research Journals Primary\_atype: Research Subject\_term:
##                   Computational biology and bioinformatics;Vaccines;Viral infection Subject\_term\_id:
##                   computational-biology-and-bioinformatics;vaccines;viral-infection}
## }

library(rjson)


rm(list=ls())


myfun_titre <- function(x,targettime, ct=250) {
  # x is the starting titre
  # ct is the cut of time after which the decay rate starts to reduce (default 250 days)

  decayRate <- rep(dr1, length(days)) # set initial decay rate
  decayRate[days>targettime]=dr2 # set reduced decay rate

  # smooth transition between decay rates
  slowing=(1/(targettime-ct))*(dr1-dr2)
  decayRate[days>ct & days<=targettime] <- dr1-slowing*(days[days>ct & days<=targettime]-ct)

  # Models decay with time
  titre_with_time <- log(x)
  for (i in 2:length(days)){
    titre_with_time[i] <- titre_with_time[i-1]+decayRate[i]
  }

  return(exp(titre_with_time))
}


ProbRemainUninfected <- function(logTitre,logk,C50){1/(1+exp(-exp(logk)*(logTitre-C50)))}

LogisticModel_PercentUninfected=function(mu_titre,sig_titre,logk,C50){

  Output<-NULL

  if (length(C50)==1) {
    C50=rep(C50,length(mu_titre))
  }

  if (length(logk)==1) {
    logk=rep(logk,length(mu_titre))
  }

  for (i in 1:length(mu_titre)) {
    Step=sig_titre[i]*0.001
    IntegralVector=seq(mu_titre[i]-5*sig_titre[i],mu_titre[i]+5*sig_titre[i],by=Step)
    Output[i]=sum(ProbRemainUninfected(IntegralVector,logk[i],C50[i])*dnorm(IntegralVector,mu_titre[i],sig_titre[i]))*Step
  }
  Output
}



mild_ef <- c(70,80,90,95) #
targetTimes<-365*c(1,1.5,2)


# Set parameter values
std10 <- 0.44 # Pooled standard deviation of antibody level on log10 data
std <- log(10^std10) # Standard deviation in natural log units
hl <- 108 # Half life of antibody decay
k <- 3.0/log(10) # logistic k value in natural log units (divided by log10 to convert it to natural logarithm units)
logk <- log(k) # log(k)

hl1 <- hl # Early half life in days
dr1 <- -log(2)/hl1 # Early corresponding decay rate (days)
hl2 <- 3650 # Late half life in days
dr2 <- -log(2)/hl2 # Late corresponding decay rate
days <- seq(0,730,1) # Total number of days to model decay with time
threshold_mild <- 0.20 # Mild EC50 value
threshold_severe <- 0.030 # Severe EC50 value
threshold_dif <- (log(threshold_mild) - log(threshold_severe))/std # EC50 difference in natural log standard errors



decay_of_mean_w_time <- log(myfun_titre(1,targetTimes[2])) # Models the decay with time using previous function



q_neut_titre_mild <- qnorm(1-mild_ef/100, 0, std) # q value for mild threshold in natural log units, used as initial guess in nlm fitting below
q_neut_titre_mild_logistic <- NULL
for (i in 1:length(mild_ef)){
  q_neut_titre_mild_logistic[i] <- -nlm(function(mu){abs(LogisticModel_PercentUninfected(mu,std,logk,0)-mild_ef[i]/100)},-q_neut_titre_mild[i])$estimate
}

mildEfficacyLogistic <- NULL
  for (j in 1:length(decay_of_mean_w_time)){
    mildEfficacyLogistic[j] <- LogisticModel_PercentUninfected(decay_of_mean_w_time[j], std, logk, q_neut_titre_mild_logistic[1])*100
  }

# write data to package
write(toJSON(mildEfficacyLogistic/100.), '../apsrm/vaccineefficacy.json')

# write to documentation
sink('../doc/source/vaccine-efficacy.rst')
cat('.. _vaccine-efficacy-data:\n')
cat('\n')
cat('Vaccine Efficacy Data\n')
cat('=====================\n')
cat('\n')
cat('.. image:: /_images/vaccine-efficacy.png\n')
cat('\n')
cat('Vaccine efficacy data is stored in the file *vaccineefficacy.json*, which contains::\n')
cat('\n    ')
cat(gsub('(\\n)', '\n    ', toJSON(mildEfficacyLogistic/100., indent=4)))
sink()
