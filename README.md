rustfst-ngram: a minimalistic WFST-based N-Gram model trainer
=============================================================

Yes, of course, N-Gram models are thoroughly obsolete, since they
cannot match the awesome bullshit-generation capacity of the latest
iteration of trillion-parameter and billion-dollar Transformers.

There is still some use in creating them, particularly when expressed
as weighted finite-state acceptors.

This is a very, very partial rewrite of [OpenGRM
NGram](https://www.openfst.org/twiki/bin/view/GRM/NGramLibrary) using
[rustfst](https://github.com/Garvys/rustfst).  It will remain very
partial, because there isn't any good reason to support anything other
than Kneser-Ney smoothing, or any arbitrary semirings, or so forth.

At the moment it only builds models in-memory, but we will implement
FAR support soon, I hope.
