# Transmitter

## Motivation

Making decisions in an environment completely dominated by uncertainty is, in strict terms, equivalent to operating without prior structure. There is no initial information that allows reducing that uncertainty, there are no known rules, and consequently, there is no criterion that allows comparing alternatives before having tried them. Even so, the system is forced to act.

Faced with this situation, the predominant approaches, especially those based on neural networks, adopt a specific strategy: introducing a deterministic structure and handling uncertainty through random initialization of their parameters. In practice, this implies fixing the behavior of the system based on an initial configuration that does not respond to any prior knowledge, but rather to an arbitrary choice.

This approach has proven effective in multiple contexts. However, its effectiveness rests on an assumption that should be made explicit: uncertainty is not resolved, but rather replaced by an initial decision that conditions the entire subsequent process.

From that point on, the system evolves within the limits imposed by that choice. This explains both its tendency to get trapped in suboptimal solutions and the difficulty of exploring alternative regions of the decision space, as well as the practical impossibility of finding optimal configurations in unbounded search spaces.

Therefore, the problem is not merely technical, but conceptual. A bias is introduced from the very beginning without having arguments to justify it, and that bias conditions the behavior of the system without offering guarantees about its validity.

From this arises the need for a different approach. Instead of replacing uncertainty with an arbitrary structure, the goal is to start from it and use it as the foundation of the decision-making process. This implies building knowledge exclusively from experience, without introducing prior assumptions that cannot be sustained.

That is the framework in which the Transmitter is defined.

## Solution

### Total symmetry

If no prior information exists, there is no valid criterion to prefer one decision over another. All alternatives are indistinguishable from the perspective of available knowledge.

From this premise, any attempt to favor some decisions over others necessarily introduces an unjustified bias. That bias does not respond to evidence, but to an arbitrary choice, and therefore conditions the behavior of the system without a rational basis.

For this reason, the only coherent strategy is to treat all decisions as equally probable. Not as a simplification, but as a direct consequence of the absence of information.

In this state, the specific decision that is made has no intrinsic meaning. It could have been any other without altering the validity of the process. What matters is not the choice itself, but the fact that there is no reason to prefer it.

In this way, the system starts from a state of complete symmetry in which no alternative stands out over the others and where that equality explicitly reflects the absence of knowledge.

### Breaking symmetry

However, this initial situation does not allow progress on its own. Every decision-making process is oriented toward achieving an objective, and without a mechanism that relates decisions to outcomes, the system cannot move forward.

That mechanism is feedback.

After each decision, the system receives a signal that evaluates how appropriate that decision has been in relation to the objective. This evaluation does not need to be exact, but it must be consistent in relative terms so that comparisons can be established.

At the moment this signal is available, the initial symmetry is no longer sustainable, since evidence appears that allows distinguishing between decisions. Consequently, the system must reflect that information.

This implies that decisions that produce better results must increase their probability of being selected in the future, while those that produce worse results must see their probability reduced.

In addition, this update must capture differences in intensity. Not all favorable decisions are equally good, and some decisions are not only ineffective, but actively hinder the achievement of the objective. The system must be able to distinguish between these cases and adjust its probabilities accordingly.

### Comparing is learning

Once feedback is introduced, learning depends on the accumulation of experience.

A single observation is not sufficient to significantly modify the behavior of the system, since it does not allow distinguishing between signal and noise. Its relevance emerges when it is integrated into a broader set of evaluations.

For this reason, the system must repeat the cycle of decision and evaluation iteratively. As it does so, the probabilities associated with each decision are progressively adjusted based on the accumulated history.

This adjustment must be gradual. If each new observation had a disproportionate impact, the system would become dominated by partial information and lose stability. By incorporating information progressively, the system builds a representation that is more consistent with the accumulated evidence.

In this context, the learning rate acts as the mechanism that regulates how much weight each new observation has relative to the set of previous experiences.

### Nature of the system

Given this update process, the Transmitter does not introduce a deterministic structure over the decision space at any point. Its behavior is probabilistic from the beginning and remains so throughout the entire process.

As a consequence, exploration is not an added element, but an inherent property. The system does not need external mechanisms to explore, since the probability distribution itself guarantees that all decisions continue to be considered, albeit with different intensities.

At the same time, decisions are not eliminated or completely discarded, but reweighted. This allows the system to continue evaluating alternatives even when there is sufficient evidence to prefer some over others.

This behavior implies that, by redistributing probability based on feedback, the system favors certain regions of the space over others. In practice, this is equivalent to generating implicit hypotheses about where it is more likely to find suitable solutions.

These hypotheses are not formulated explicitly, but are encoded in the probabilistic structure of the system, which continuously adjusts based on experience.

## Conclusion

The Transmitter addresses decision-making under uncertainty without introducing arbitrary assumptions or structures imposed from the outset.

It starts from a situation of complete symmetry, incorporates information only when it becomes available, and progressively adjusts its behavior based on accumulated experience.

In this way, the system does not replace uncertainty, but operates within it using exclusively the information it obtains throughout the process.

## Example

Imagine you wake up on a deserted island. There is no one else, you cannot communicate with anyone and, after exploring a bit, you find a single place where you can get supplies: a supermarket.

You walk in expecting to find something familiar, but you quickly realize the problem. None of the products are recognizable. There are no labels you can understand, you do not know what ingredients they contain, how they taste, or even how they will affect you.

You know you will have to manage there for a long time, probably months, so eating every day is not optional.

Under these conditions, the first decision is unavoidable: what do you choose to eat today? You might be tempted to rely on appearance, to think that one dish looks appealing or another seems lighter, but if you stop for a moment you will see that all of that is just unsupported guesswork. You are introducing criteria you cannot justify. You do not know what each dish contains, you do not know how it will affect you and, strictly speaking, you know nothing.

Therefore, if you are rigorous, you cannot defend that one dish is better than another. And if you cannot defend it, any preference you introduce will be an arbitrary bias. The only coherent strategy, then, is to treat all options as equally probable and choose at random.

Suppose you pick a random dish. You eat it and, for the first time, you obtain information: you like it and it sits well with you. You do not know if it is the best dish in the supermarket, but you do know it is not a bad choice. That is all the evidence you have.

The next day you face the same problem again. However, you are no longer exactly at the same point: you now have a prior observation. Does it make sense to ignore it and choose completely at random again? Intuitively, no. But you still cannot conclude that the dish is optimal: you have only tried it once.

Thus, an inevitable tension appears between what you know and what you still do not know. The coherent way to resolve it is not to decide deterministically, but to slightly adjust your preferences: increase the probability of choosing something similar to what has worked, without discarding the rest of the options.

In the next choice, therefore, you no longer start from a completely uniform distribution. It is more likely that you will repeat something similar, although you can still choose any other dish. Suppose you try a different one and the result is worse. Now you do not only have positive evidence, but also comparison: you know that one option has been better than another.

From this point on, the process gains continuity. Each day you choose based on what you already know, each day you obtain feedback and each day you adjust your preferences incrementally. If certain dishes consistently work well, they will progressively gain weight; if others fail, they will lose it.

However, this adjustment is never absolute. Even if some dishes become clearly preferable, no option disappears completely. There is always a probability, however small, of trying something different. This is not a minor detail: it is what allows you to avoid closing prematurely on a reduced set of options.

In fact, if you consider what would happen if the best dish were in a region you have barely explored, the answer is clear: you could only discover it if you keep open the possibility of returning to that region. And since you never eliminate any option completely, you retain that capability.

Over time, your choices stop being arbitrary, but they do not become deterministic either. What emerges is a structure of preferences that reflects everything you have experienced so far: you choose with higher probability what has worked better, without completely giving up exploration.

That behavior is, in essence, that of the Transmitter: starting from ignorance, incorporating evidence progressively and continuously reorganizing decision-making based on experience.
