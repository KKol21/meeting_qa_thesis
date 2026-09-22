# Result visualizations

Run `python src/tools/plot_results.py` to recreate every figure as both PNG and
vector PDF under `docs/figures/results`. The script reads saved JSON artifacts;
it does not recompute experimental results.

## Main results

1. **Test retrieval trade-off** ([PNG](figures/results/01-test-retrieval-tradeoff.png) · [PDF](figures/results/01-test-retrieval-tradeoff.pdf)). Precision and
   recall are shown jointly for every chunker, retriever, and evidence budget.
   The connected points progress from 512 to 1,024 to 2,048 words (right to left), making the budget trade-off
   visible without collapsing matched environments.

2. **Test downstream performance** ([PNG](figures/results/02-test-downstream-performance.png) · [PDF](figures/results/02-test-downstream-performance.pdf)).
   Meeting-macro ROUGE-L, BERTScore F1, and LLM-judge means with 95% paired
   meeting-cluster bootstrap intervals. This is the clearest overview of why
   retrieval differences do not translate into similarly large answer gains.

3. **Paired Lumber effects** ([PNG](figures/results/03-test-lumber-paired-effects.png) · [PDF](figures/results/03-test-lumber-paired-effects.pdf)). Lumber minus
   turn-packed and Lumber minus word-packed within each matched retriever and
   budget. The zero line and confidence intervals distinguish directional
   differences from uncertain effects. This should accompany the headline
   average Lumber effect in the text.

4. **Budget-change associations** ([PNG](figures/results/04-test-budget-proportionality.png) · [PDF](figures/results/04-test-budget-proportionality.pdf)). Question-level
   changes from 1,024 to 2,048 words for Lumber+dense, with fixed bin means
   and meeting-cluster bootstrap intervals overlaid. The left column relates
   recall gain to answer changes; the right column relates precision loss to
   answer changes. Points for the ordinal judge are vertically jittered only
   for display. These associations do not identify a causal mechanism.

5. **Oracle model and evidence comparison** ([PNG](figures/results/05-test-oracle-models.png) · [PDF](figures/results/05-test-oracle-models.pdf)). The first
   point uses retrieved evidence capped at 1,024 words with the 14B answer
   model; the remaining points use uncapped annotated evidence with the 7B,
   14B, and quantized 32B models. The retrieved condition is the predeclared
   `lumber__dense__w1024` comparison. The gap changes both evidence content
   and available context length.

## Validation and secondary analyses

6. **Lumber target selection** ([PNG](figures/results/06-validation-lumber-target.png) · [PDF](figures/results/06-validation-lumber-target.pdf)). Shows the
   retrieval criterion, zero-hit rate, and chunk geometry that motivated the
   1,000 pseudo-token target. The shaded recall band marks values within 0.01 of
   the best validation mean; dashed lines show deterministic-baseline medians.

7. **Deterministic chunk-size sensitivity**
   ([PNG](figures/results/07-validation-baseline-size.png) · [PDF](figures/results/07-validation-baseline-size.pdf)). Cell-level heatmaps show recall and F1
   changes from the 256-word reference without averaging over the correlated
   retriever-by-budget environments.

8. **Boundary-shuffled control** ([PNG](figures/results/08-validation-boundary-control.png) · [PDF](figures/results/08-validation-boundary-control.pdf)). Paired
   recall and F1 effects compare actual Lumber boundaries with geometry-matched,
   semantically meaningless boundaries. Use this as the direct test of whether
   boundary placement itself matters.

9. **Clipping-policy sensitivity** ([PNG](figures/results/09-validation-clipping-policy.png) · [PDF](figures/results/09-validation-clipping-policy.pdf)). Heatmaps
   show retrieval changes from dropping or expanding the final partial chunk,
   relative to the main clipping policy. This is retrieval-only and belongs in
   the secondary analysis or appendix.

10. **Boundary-model diagnostic** ([PNG](figures/results/10-validation-boundary-model.png) · [PDF](figures/results/10-validation-boundary-model.pdf)). Mean recall,
    chunk geometry, and boundary agreement for the three boundary models. It is
    based on only five meetings and should be reported as a failure check, not
    evidence of model equivalence.

First-overlap MRR is intentionally not promoted to a main figure because chunk
size mechanically changes overlap probability. Its values remain in the result
tables. The qualitative workbook is also not plotted yet because its manual
annotation fields have not been completed.
