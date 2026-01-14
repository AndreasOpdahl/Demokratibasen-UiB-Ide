"""
Metrics for general capability retention evaluation (both in-training validation and final evaluation) 
of summarisation models for Norwegian public documents, assuming that LLM-generated reference summaries are available.

TODO: Below follows the ambition - not yet implemented.

The evaluation regime measures:
* 
* Regression vs the base model
* Distributional drift
* Trade-offs between specialization and generality

Types of metrics:
* Norwegian general-domain NLL / perplexity (cheap early signal)
  * text-prediction on fixed corpus
* Base-vs-tuned divergence (early warning): measures distributional drift via token-level KL divergence or log-prob deltas
  * Norwegian anchor prompts (general, not summarisation)
* Anchor-suite retention (primary signal): mean and worst-case retention of general capability after specialization
  * Norwegian reading comprehension / QA
  * General instruction following (Norwegian)
  * Simple reasoning / logic in Norwegian
  * Language modeling probes (cloze / continuation)
  * ...compare delta = tuned - base scores
  
"""



