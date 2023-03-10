# Hidden Markov Models
Implementation solutions to two problems associated with HMMs

The Viterbi algorithm is used for supervised tasks and the Forward-Backward algorithm is employed for semi-supervised, and unsupervised tasks.

A hidden Markov model (HMM) allows us to talk about both observed events (e.g. words) and hidden events (e.g. part-of-speech tags). An HMM is specified by the following components:

![image](https://user-images.githubusercontent.com/24508376/219600466-3be87292-2a43-4ccb-9df7-a873e32c9d8f.png)

Where πππ = π·π(ππ+π = πΊπ|ππ = πΊπ) = (ππ+π = ππππ|ππ = ππππ), and πππ = π·π(πΆπ = π|ππ = πΊπ) = π·π(πΆπ = πππππ|ππ = ππππ).

# The Viterbi Algorithm (supervised task)
For any model, such as an HMM, that contains hidden variables, the task of finding which sequence of variables is the most likely tag sequence given the sequence of observations (words), is called the decoding task. The task of the decoder is to find the best hidden variable sequence (π1π2π3 β¦ ππ). The most common decoding algorithms for HMMs is the Viterbi algorithm. This algorithm is a kind of dynamic programming.

Each cell ππ( π), represents the probability that the HMM is in state π after seeing the first π observations and passing through the most probable state sequence π1 β¦ ππ‘β1. The value of each cell ππ( π) is computed by recursively taking the most probable path. Like other dynamic programming algorithms, Viterbi fills each cell recursively. The Viterbi probability is computed by taking the most probable of the extensions of the paths that lead to the current cell, provided the Viterbi probability had already been calculated in every state at time π‘ β 1. For a given state ππ at time t, the Viterbi probability ππ( π) is computed in log space as:

            π
π£π‘ (π) β   πππ₯ ( π£π‘β1(π) + ln (πππ ) + l n (ππ(ππ‘ )))
          π = 1
    
    
Where π£π‘β1(π) is the previous Viterbi path probability from the previous time step, πππ is the transition probability from previous state ππ to the current state ππ, and ππ(ππ‘ ) is the emission probability of the observation symbol ππ‘ given the current state π. Pseudocode for the Viterbi algorithm is given in the following.

Function VITERBI (observations of len T, state-graph of len N) returns best-path
create a path probability matrix π£ππ‘ππππ[π, π]
for each state s from 1 to N do // Initialization step
  π£ππ‘ππππ[π , 1] β ln (ππ  ) + ln (ππ  (π1))
  πππππππππ‘ππ[π , 1] β 0
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
                                   π
    π£ππ‘ππππ[π , π‘] β l n(ππ  (ππ‘ )) + πππ₯ ( π£ππ‘ππππ[π β², π‘ β 1] + ln (ππ β²,π  ) )
                                π β² = 1
   
                         π
    πππππππππ‘ππ[π , π‘] β ππππππ₯ π£ππ‘ππππ[π β², π‘ β 1] + ln (ππ β²,π  ))
                      π β² = 1

                    π
  πππ π‘πππ‘βπππππ‘ππ β  ππππππ₯ π£ππ‘ππππ[π , π] // termination step
                  π  = 1
πππ π‘πππ‘β β π‘βπ πππ‘β π π‘πππ‘πππ ππ‘ π π‘ππ‘π πππ π‘πππ‘βππππ‘ππ, π‘βπ ππππππ€π  πππππππππ‘ππ[ ]π‘π π π‘ππ‘ππ  ππππ ππ π‘πππ

return πππ π‘πππ‘β

# The Forward-Backward Algorithm (semi-supervised task)
This algorithm learns the parameters of an HMM, which are, the transition probability matrix A, and the emission probability matrix B in a semi-supervised manner. In fact, the input to such a learning algorithm would be an unlabeled sequence of observations O and a vocabulary of potential hidden states Q.

The standard algorithm for HMM training is the forward-backward, or Baum-Welch algorithm, a special case of the Expectation-Maximization or EM algorithm. The algorithm trains both the transition probabilities A and the emission probabilities B of the HMM. EM is an iterative algorithm, computing an initial estimate for the probabilities, then using those estimates to computing a better estimate, and so on, iteratively improving the probabilities that it learns. The Baum-Welch algorithm solves this problem by iteratively estimating the counts. The Baum-Welch algorithm starts with an estimate for the transition and observation probabilities and then uses these estimated probabilities to derive better and better probabilities.



To understand the algorithm, we need to define the forward and backward probabilities. The forward algorithm is a kind of dynamic programming algorithm, that is, an algorithm that uses a table to store intermediate values as it builds up the probability of the observation sequence. The forward algorithm computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence.
Each cell of the forward algorithm at πΆπ(π) represents the probability of being in state π after seeing the first π observations. The value of each cell at πΆπ(π) is computed by summing over the probabilities of every path that could lead to this cell. Formally, each cell expresses the following probability:

πΌπ‘ (π) = π(π1, π2, β¦ ππ‘ . , ππ‘ = π)

Here, ππ = π means βthe πππ state in the sequence of states is state πβ. This probability at πΆπ(π) is computed by summing over the extensions of all the paths that lead to the current cell. For a given state ππ at time t, the value at πΆπ(π) is computed as
         π
πΌπ‘ (π) = Ξ£ πΌπ‘β1(π)πππππ(ππ‘)
        π=1
Where πΌπ‘β1(π) is the previous forward path probability from the previous time step. The pseudocode for the forward algorithm is given in the following.

Function ForwardAlg1 (observations of len T, state-graph of len N) returns forward-prob
create a probability matrix ππππ€πππ[π, π]
for each state s from 1 to N do // Initialization step
  ππππ€πππ[π , 1] β ππ  β ππ  (π1)
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
                  π
  ππππ€πππ[π , π‘] β Ξ£ ππππ€πππ[π β², π‘ β 1]
                π β²=1
return ππππ€πππ


There are some implementational issues both for the Forward algorithm and the Backward algorithm described later. The most severe practical problem is that multiplying many probabilities always yields very small numbers that will give underflow errors on any computer. For this reason, the Forward algorithm has been presented by the ForwardAlg2 done in log space, which will make the numbers stay reasonable.

Function ForwardAlg2 (observations of len T, state-graph of len N) returns forward-prob
create a probability matrix ππππ€πππ[π, π]
for each state s from 1 to N do // Initialization step
  ππππ€πππ[π , 1] β ln ( ππ  ) + ln (ππ  (π1))
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
    π‘ππ = ππππ€πππ[1, π‘ β 1] + ln(π1π  ) + ππ(ππ  (ππ‘ ))
    for each state π β² from 2 to N
      π‘ππ1 = ππππ€πππ[π β², π‘ β 1] + ln(ππ β²π  ) + ππ(ππ  (ππ‘ ))
               {tmp + ln (1 + exp (π‘ππ1 β π‘ππ))) ππ π‘ππ1 β€ π‘ππ
        π‘ππ β {
               {tmp1 + ln (1 + exp (π‘ππ β π‘ππ1))) π, π€.
      ππππ€πππ[π , π‘] = π‘ππ
return ππππ€πππ


The backward probability π· is the probability of seeing the observations from time π + π to the end, given in state π at time π‘: π·π(π) = π·(ππ+π, ππ+π, β¦ , ππ»|ππ = π).


