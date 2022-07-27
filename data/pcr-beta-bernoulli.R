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

library(rjson)
library(fitdistrplus)
library(ggplot2)
library(egg)
library(gridExtra)

posterior <- readRDS('pcr-sensitivity.rds')
bestFits <- t(sapply(apply(posterior, 2, fitdist, 'beta'), '[[', 'estimate'))
colnames(bestFits) <- c('alpha', 'beta')


# write as json to package
write(toJSON(as.data.frame(bestFits), indent=4), '../apsrm/pcrbetaparams.json')


# write data and text to documentation
sink('../doc/source/pcr-test-data.rst')
cat('.. _pcr-test-data:\n')
cat('\n')
cat('PCR Test Data\n')
cat('=============\n')
cat('\n')
cat('PCR Test data is stored in the file *pcrbetaparams.json* and looks like the following:\n')
cat('\n')
cat('.. figure:: /_images/pcr-beta-dists.png\n')
cat('\n')
cat('    Distributions of the daily senstitvities of the PCR test. Bars show the raw data and densities shown the fitted (Beta) distributions.\n')
cat('\n')
cat('.. figure:: /_images/pcr-daily-means.png\n')
cat('\n')
cat('    Mean daily sensitivities.\n')
cat('\n')
cat('The beta parameters are::\n')
cat('\n    ')
cat(gsub('(\\n)', '\n    ', toJSON(as.data.frame(bestFits), indent=4)))
sink()


# write graphics to documenation
bestFits <- sapply(apply(posterior, 2, fitdist, 'beta'), '[[', 'estimate')
mns <- apply(bestFits, 2, function(p) p[1] / sum(p))

# the daily averages
png('../doc/source/_images/pcr-daily-means.png')
plot(ggplot(data.frame(x=1:ncol(bestFits), y=mns), aes(x=x, y=y)) +
     ylim(c(0., 1.)) + geom_point() + xlab('Day') + ylab('Sensitivity') +
     theme(text=element_text(size=17)))
dev.off()

# the daily beta distributions
png('../doc/source/_images/pcr-beta-dists.png', width=1000, height=1500)
plots <- lapply(1:ncol(bestFits), function(col) {
    params <- bestFits[,col]
    dat <- data.frame(x=posterior[,col])
    ggplot(data.frame(x = c(0., 1.)), aes(x = x)) +
      xlim(c(0., 1.)) +
      geom_histogram(mapping=aes(x=x, y=..density..), data=dat, binwidth=.05) +
      stat_function(
        fun = dbeta,
        args = list(shape1 = params[1], shape2 = params[2]),
        geom = "area",
        fill = "green",
        alpha = 0.25) +
      stat_function(
        fun = dbeta,
        args = list(shape1 = params[1], shape2 = params[2])) +
      labs(
        x = "p",
        y = paste0('beta(p; ', sprintf('%0.2f', params[1]), ', ', sprintf('%0.2f', params[2]), ')'),
        subtitle = paste('day', col)) +
      theme(
        text=element_text(size=17),
        plot.title = element_text(hjust = 0.5))
})
grid.arrange(grobs=plots, ncol=3)
dev.off()
