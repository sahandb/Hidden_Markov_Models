# Hidden Markov Models
Implementation solutions to two problems associated with HMMs

The Viterbi algorithm is used for supervised tasks and the Forward-Backward algorithm is employed for semi-supervised, and unsupervised tasks.

A hidden Markov model (HMM) allows us to talk about both observed events (e.g. words) and hidden events (e.g. part-of-speech tags). An HMM is specified by the following components:

![image](https://user-images.githubusercontent.com/24508376/219600466-3be87292-2a43-4ccb-9df7-a873e32c9d8f.png)

Where 𝒂𝒊𝒋 = 𝑷𝒓(𝒒𝒕+𝟏 = 𝑺𝒋|𝒒𝒕 = 𝑺𝒊) = (𝒒𝒕+𝟏 = 𝒕𝒂𝒈𝒋|𝒒𝒕 = 𝒕𝒂𝒈𝒊), and 𝒃𝒋𝒌 = 𝑷𝒓(𝑶𝒕 = 𝒌|𝒒𝒕 = 𝑺𝒋) = 𝑷𝒓(𝑶𝒕 = 𝒘𝒐𝒓𝒅𝒌|𝒒𝒕 = 𝒕𝒂𝒈𝒋).

# The Viterbi Algorithm (supervised task)
For any model, such as an HMM, that contains hidden variables, the task of finding which sequence of variables is the most likely tag sequence given the sequence of observations (words), is called the decoding task. The task of the decoder is to find the best hidden variable sequence (𝑞1𝑞2𝑞3 … 𝑞𝑛). The most common decoding algorithms for HMMs is the Viterbi algorithm. This algorithm is a kind of dynamic programming.

Each cell 𝒗𝒕( 𝒋), represents the probability that the HMM is in state 𝒋 after seeing the first 𝒕 observations and passing through the most probable state sequence 𝑞1 … 𝑞𝑡−1. The value of each cell 𝒗𝒕( 𝒋) is computed by recursively taking the most probable path. Like other dynamic programming algorithms, Viterbi fills each cell recursively. The Viterbi probability is computed by taking the most probable of the extensions of the paths that lead to the current cell, provided the Viterbi probability had already been calculated in every state at time 𝑡 − 1. For a given state 𝒒𝒋 at time t, the Viterbi probability 𝒗𝒕( 𝒋) is computed in log space as:

            𝑁
𝑣𝑡 (𝑗) ←   𝑚𝑎𝑥 ( 𝑣𝑡−1(𝑖) + ln (𝑎𝑖𝑗 ) + l n (𝑏𝑗(𝑜𝑡 )))
          𝑖 = 1
    
    
Where 𝑣𝑡−1(𝑖) is the previous Viterbi path probability from the previous time step, 𝑎𝑖𝑗 is the transition probability from previous state 𝑞𝑖 to the current state 𝑞𝑗, and 𝑏𝑗(𝑜𝑡 ) is the emission probability of the observation symbol 𝑜𝑡 given the current state 𝑗. Pseudocode for the Viterbi algorithm is given in the following.

Function VITERBI (observations of len T, state-graph of len N) returns best-path
create a path probability matrix 𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑁, 𝑇]
for each state s from 1 to N do // Initialization step
  𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑠, 1] ← ln (𝜋𝑠 ) + ln (𝑏𝑠 (𝑜1))
  𝑏𝑎𝑐𝑘𝑝𝑜𝑖𝑛𝑡𝑒𝑟[𝑠, 1] ← 0
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
                                   𝑁
    𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑠, 𝑡] ← l n(𝑏𝑠 (𝑜𝑡 )) + 𝑚𝑎𝑥 ( 𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑠′, 𝑡 − 1] + ln (𝑎𝑠′,𝑠 ) )
                                𝑠′ = 1
   
                         𝑁
    𝑏𝑎𝑐𝑘𝑝𝑜𝑖𝑛𝑡𝑒𝑟[𝑠, 𝑡] ← 𝑎𝑟𝑔𝑚𝑎𝑥 𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑠′, 𝑡 − 1] + ln (𝑎𝑠′,𝑠 ))
                      𝑠′ = 1

                    𝑁
  𝑏𝑒𝑠𝑡𝑝𝑎𝑡ℎ𝑝𝑜𝑖𝑛𝑡𝑒𝑟 ←  𝑎𝑟𝑔𝑚𝑎𝑥 𝑣𝑖𝑡𝑒𝑟𝑏𝑖[𝑠, 𝑇] // termination step
                  𝑠 = 1
𝑏𝑒𝑠𝑡𝑝𝑎𝑡ℎ ← 𝑡ℎ𝑒 𝑝𝑎𝑡ℎ 𝑠𝑡𝑎𝑟𝑡𝑖𝑛𝑔 𝑎𝑡 𝑠𝑡𝑎𝑡𝑒 𝑏𝑒𝑠𝑡𝑝𝑎𝑡ℎ𝑜𝑖𝑛𝑡𝑒𝑟, 𝑡ℎ𝑒 𝑓𝑜𝑙𝑙𝑜𝑤𝑠 𝑏𝑎𝑐𝑘𝑝𝑜𝑖𝑛𝑡𝑒𝑟[ ]𝑡𝑜 𝑠𝑡𝑎𝑡𝑒𝑠 𝑏𝑎𝑐𝑘 𝑖𝑛 𝑡𝑖𝑚𝑒

return 𝑏𝑒𝑠𝑡𝑝𝑎𝑡ℎ

# The Forward-Backward Algorithm (semi-supervised task)
This algorithm learns the parameters of an HMM, which are, the transition probability matrix A, and the emission probability matrix B in a semi-supervised manner. In fact, the input to such a learning algorithm would be an unlabeled sequence of observations O and a vocabulary of potential hidden states Q.

The standard algorithm for HMM training is the forward-backward, or Baum-Welch algorithm, a special case of the Expectation-Maximization or EM algorithm. The algorithm trains both the transition probabilities A and the emission probabilities B of the HMM. EM is an iterative algorithm, computing an initial estimate for the probabilities, then using those estimates to computing a better estimate, and so on, iteratively improving the probabilities that it learns. The Baum-Welch algorithm solves this problem by iteratively estimating the counts. The Baum-Welch algorithm starts with an estimate for the transition and observation probabilities and then uses these estimated probabilities to derive better and better probabilities.



To understand the algorithm, we need to define the forward and backward probabilities. The forward algorithm is a kind of dynamic programming algorithm, that is, an algorithm that uses a table to store intermediate values as it builds up the probability of the observation sequence. The forward algorithm computes the observation probability by summing over the probabilities of all possible hidden state paths that could generate the observation sequence.
Each cell of the forward algorithm at 𝜶𝒕(𝒋) represents the probability of being in state 𝒋 after seeing the first 𝒕 observations. The value of each cell at 𝜶𝒕(𝒋) is computed by summing over the probabilities of every path that could lead to this cell. Formally, each cell expresses the following probability:

𝛼𝑡 (𝑗) = 𝑃(𝑜1, 𝑜2, … 𝑜𝑡 . , 𝑞𝑡 = 𝑗)

Here, 𝒒𝒕 = 𝒋 means “the 𝒕𝒕𝒉 state in the sequence of states is state 𝒋”. This probability at 𝜶𝒕(𝒋) is computed by summing over the extensions of all the paths that lead to the current cell. For a given state 𝒒𝒋 at time t, the value at 𝜶𝒕(𝒋) is computed as
         𝑁
𝛼𝑡 (𝑗) = Σ 𝛼𝑡−1(𝑖)𝑎𝑖𝑗𝑏𝑗(𝑜𝑡)
        𝑖=1
Where 𝛼𝑡−1(𝑗) is the previous forward path probability from the previous time step. The pseudocode for the forward algorithm is given in the following.

Function ForwardAlg1 (observations of len T, state-graph of len N) returns forward-prob
create a probability matrix 𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑁, 𝑇]
for each state s from 1 to N do // Initialization step
  𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠, 1] ← 𝜋𝑠 ∗ 𝑏𝑠 (𝑜1)
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
                  𝑁
  𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠, 𝑡] ← Σ 𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠′, 𝑡 − 1]
                𝑠′=1
return 𝑓𝑜𝑟𝑤𝑎𝑟𝑑


There are some implementational issues both for the Forward algorithm and the Backward algorithm described later. The most severe practical problem is that multiplying many probabilities always yields very small numbers that will give underflow errors on any computer. For this reason, the Forward algorithm has been presented by the ForwardAlg2 done in log space, which will make the numbers stay reasonable.

Function ForwardAlg2 (observations of len T, state-graph of len N) returns forward-prob
create a probability matrix 𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑁, 𝑇]
for each state s from 1 to N do // Initialization step
  𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠, 1] ← ln ( 𝜋𝑠 ) + ln (𝑏𝑠 (𝑜1))
for each time step t from 2 to T do // recursion step
  for each state s from 1 to N do
    𝑡𝑚𝑝 = 𝑓𝑜𝑟𝑤𝑎𝑟𝑑[1, 𝑡 − 1] + ln(𝑎1𝑠 ) + 𝑙𝑛(𝑏𝑠 (𝑜𝑡 ))
    for each state 𝑠′ from 2 to N
      𝑡𝑚𝑝1 = 𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠′, 𝑡 − 1] + ln(𝑎𝑠′𝑠 ) + 𝑙𝑛(𝑏𝑠 (𝑜𝑡 ))
               {tmp + ln (1 + exp (𝑡𝑚𝑝1 − 𝑡𝑚𝑝))) 𝑖𝑓 𝑡𝑚𝑝1 ≤ 𝑡𝑚𝑝
        𝑡𝑚𝑝 ← {
               {tmp1 + ln (1 + exp (𝑡𝑚𝑝 − 𝑡𝑚𝑝1))) 𝑜, 𝑤.
      𝑓𝑜𝑟𝑤𝑎𝑟𝑑[𝑠, 𝑡] = 𝑡𝑚𝑝
return 𝑓𝑜𝑟𝑤𝑎𝑟𝑑


The backward probability 𝜷 is the probability of seeing the observations from time 𝒕 + 𝟏 to the end, given in state 𝑖 at time 𝑡: 𝜷𝒕(𝒊) = 𝑷(𝒐𝒕+𝟏, 𝒐𝒕+𝟐, … , 𝒐𝑻|𝒒𝒕 = 𝒊).


