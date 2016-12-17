# onlineldavb.py: Package of functions for fitting Latent Dirichlet
# Allocation (LDA) with online variational Bayes (VB).
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
meanchangethresh = 0.001


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

#change
def beta_expectation(a, c):
    """
    For a vector  x ~ Beta(a, b), computes E[log(x)] given a and b.
    Here x is the topic-word probabilities, beta.
    c = a + b
    """
    return psi(a) - psi(c)


class OnlineLDA:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K, D, alpha, eta, zeta, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A word2id mapping.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta and zeta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = vocab

        self._K = K
        self._W = len(self._vocab)
        self._D = D
        #
        self._alpha = alpha
        #change
        # beta_kw ~ beta(eta, zeta)
        self._eta = eta
        self._zeta = zeta
        #
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0

        # Initialize the variational distribution q(beta|lambda)
        #change
        #self._lambda = n.zeros((self._K, self._W)) + 0.0001 #
        self._lambda = 1*n.random.gamma(100., 1./100., (self._K, self._W))
        #TODO n.random.beta(eta, zeta
        #1*n.random.gamma(100., 1./100., (self._K, self._W))
        #self._mu =  n.ones((self._K, self._W))#
        self._mu = 1*n.random.gamma(100., 1./100., (self._K, self._W))

        print("lambda", self._lambda[0])
        print("mu", self._mu[0])

        #
        #self._Elogbeta = dirichlet_expectation(self._lambda)
        #change
        self._posElogbeta = beta_expectation(self._lambda, self._lambda + self._mu) # K X W
        self._posExpElogbeta = n.exp(self._posElogbeta)
        #
        self._negElogbeta = beta_expectation(self._mu, self._lambda + self._mu) # K X W
        self._negExpElogbeta = n.exp(self._negElogbeta)
        #
        print("E[log beta]", self._posElogbeta[0][1:10])
        print("exp(E[log beta])", self._posExpElogbeta[0][1:10])

        print("E[log (1-beta)]", self._negElogbeta[0][1:10])
        print("exp E[log (1-beta)]", self._negExpElogbeta[0][1:10])

    def do_e_step(self, pos_wordids, pos_wordcts, neg_wordids, neg_wordcts):
        batchD = len(pos_wordids)

        # Initialize the variational distribution q(theta|gamma) for the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        pos_sstats = n.zeros(self._lambda.shape)
        neg_sstats = n.zeros(self._lambda.shape)

        # Now, for each document d update that document's gamma and phi
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            pos_ids = pos_wordids[d]
            pos_cts = pos_wordcts[d]
            #
            neg_ids = neg_wordids[d]
            neg_cts = neg_wordcts[d]
            #
            #print("pos ids %d, neg ids %d" % (len(pos_ids), len(neg_ids)))
            #print("pos cts %d, neg cts %d" % (sum(pos_cts), sum(neg_cts)))

            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            #
            # topics
            pos_expElogbetad = self._posExpElogbeta[:, pos_ids]
            neg_expElogbetad = self._negExpElogbeta[:, neg_ids]

            # The optimal phi_{dwk} is proportional to
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            # 1 X W+
            pos_phinorm = n.dot(expElogthetad, pos_expElogbetad) + 1e-100
            # 1 X W-
            neg_phinorm = n.dot(expElogthetad, neg_expElogbetad) + 1e-100
            #
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                # alpha + 1 X K * {(1 X W+) X (W+ X K) + }  --> 1 X K TODO
                gammad = self._alpha + expElogthetad * \
                        (n.dot(pos_cts / pos_phinorm, pos_expElogbetad.T) + \
                        n.dot(neg_cts / neg_phinorm, neg_expElogbetad.T))
                #TODO check gammad
                #gammad = self._alpha + expElogthetad * n.dot(pos_cts / pos_phinorm, pos_expElogbetad.T)
                #print("gammad", gammad[:, n.newaxis])
                #
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                #
                #change
                pos_phinorm = n.dot(expElogthetad, pos_expElogbetad) + 1e-100
                neg_phinorm = n.dot(expElogthetad, neg_expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    print("converged %d" % d)
                    break
            gamma[d, :] = gammad
            #
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            # change
            pos_sstats[:, pos_ids] += n.outer(expElogthetad.T, pos_cts/pos_phinorm)
            neg_sstats[:, neg_ids] += n.outer(expElogthetad.T, neg_cts/neg_phinorm)
        #
        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
        pos_sstats = pos_sstats * self._posExpElogbeta
        neg_sstats = neg_sstats * self._negExpElogbeta

        #print("pos stats",  pos_sstats[:, pos_ids])
        #print("neg stats",  neg_sstats[:, neg_ids])

        return (gamma, pos_sstats, neg_sstats)


    def update_lambda(self, pos_wordids, pos_wordcts, neg_wordids, neg_wordcts):
        """
        First does an E step on the mini-batch given in wordids and
        wordcts, then uses the result of that E step to update the
        variational parameter matrix lambda.

        Arguments:
        docs:  List of D documents. Each document must be represented
               as a string. (Word order is unimportant.) Any
               words not in the vocabulary will be ignored.

        Returns gamma, the parameters to the variational distribution
        over the topic weights theta for the documents analyzed in this
        update.

        Also returns an estimate of the variational bound for the
        entire corpus for the OLD setting of lambda based on the
        documents passed in. This can be used as a (possibly very
        noisy) estimate of held-out likelihood.
        """

        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        # Do an E step to update gamma, phi | lambda for this
        # mini-batch. This also returns the information about phi that
        # we need to update lambda.
        (gamma, pos_sstats, neg_sstats) = self.do_e_step(pos_wordids, pos_wordcts, neg_wordids, neg_wordcts)
        # Estimate held-out likelihood for current values of lambda.
        #change
        bound = self.approx_bound(pos_wordids, pos_wordcts, neg_wordids, neg_wordcts, gamma)
        # Update lambda based on documents.
        # change
        self._lambda = self._lambda * (1-rhot) + rhot * (self._eta + self._D * pos_sstats / len(pos_wordids))
        #
        print("updated lambda",self._lambda[0][1:10])
        self._mu =  self._mu  * (1-rhot) + rhot * (self._zeta + self._D * neg_sstats / len(neg_wordids))
        print("updated mu",self._mu[0][1:10])
        #
        self._posElogbeta = beta_expectation(self._lambda, self._lambda + self._mu)
        self._posExpElogbeta = n.exp(self._posElogbeta)
        #
        self._negElogbeta = beta_expectation(self._mu, self._lambda + self._mu)
        self._negExpElogbeta = n.exp(self._negElogbeta)
        #
        self._updatect += 1

        return(gamma, bound)

    def approx_bound(self, pos_wordids, pos_wordcts, neg_wordids, neg_wordcts, gamma):
        """
        Estimates the variational bound over *all documents* using only
        the documents passed in as "docs." gamma is the set of parameters
        to the variational distribution q(theta) corresponding to the
        set of documents passed in.

        The output of this function is going to be noisy, but can be
        useful for assessing convergence.
        """

        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        batchD = len(pos_wordids)

        score = 0
        Elogtheta = dirichlet_expectation(gamma)
        #expElogtheta = n.exp(Elogtheta)

        # E[log p(docs | theta, beta)]
        for d in range(0, batchD):
            #gammad = gamma[d, :]
            # Positive words
            pos_ids = pos_wordids[d]
            pos_cts = n.array(pos_wordcts[d])
            pos_phinorm = n.zeros(len(pos_ids))
            for i in range(0, len(pos_ids)):
                temp = Elogtheta[d, :] + self._posElogbeta[:, pos_ids[i]]
                tmax = max(temp)
                pos_phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(pos_cts * pos_phinorm)
            #
            # Negative words
            neg_ids = neg_wordids[d]
            neg_cts = n.array(neg_wordcts[d])
            neg_phinorm = n.zeros(len(neg_ids))
            for i in range(0, len(neg_ids)):
                temp = Elogtheta[d, :] + self._negElogbeta[:, neg_ids[i]]
                tmax = max(temp)
                neg_phinorm[i] = n.log(sum(n.exp(temp - tmax))) + tmax
            score += n.sum(neg_cts * neg_phinorm)

        # E[log p(theta | alpha) - log q(theta | gamma)]
        score += n.sum((self._alpha - gamma) * Elogtheta)
        score += n.sum(gammaln(gamma) - gammaln(self._alpha))
        score += sum(gammaln(self._alpha * self._K) - gammaln(n.sum(gamma, 1)))

        # Compensate for the subsampling of the population of documents
        score = score * self._D / (len(pos_wordids) + len(neg_wordids))

        # E[log p(beta | eta) - log q (beta | lambda)]
        score = score + n.sum((self._eta - self._lambda) * self._posElogbeta)
        score = score + n.sum((self._zeta - self._mu) * self._negElogbeta)
        #
        score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta))
        score = score + n.sum(gammaln(self._mu) - gammaln(self._zeta))
        #
        score = score + n.sum(gammaln(self._eta + self._zeta) -
                              gammaln(self._lambda + self._mu))


        #score = score + n.sum(gammaln(self._eta * self._W) -gammaln(n.sum(self._lambda, 1)))

        return(score)

