# User-approved title-only dropout

After discussing the missing WANDS facets, the user explicitly approved proceeding
with title-only protection. No facet labels are inferred. Every other field is
eligible for independent 0.15 dropout during PI-FT training; test dropout is zero.

This amendment was applied before any Adapted-PI-FT training began. Standard-FT
was training without any permutation or dropout; its input texts, training pairs,
sample order and objective are unaffected by the protection policy. Its in-memory
trace may record the former, unused protection sets; `augment=False` means these
sets had no effect. No trained PI-FT checkpoint has been replaced or discarded.

The prior config/protocol/triples are preserved under `archive/before_title_only`.
The only triple changes are the `protect` lists, now exactly `["title"]`. Query,
positive, negatives, splits, seeds, epochs and optimizer are unchanged. Current
config hashes identify the amended files. This is an adapted experiment without
facet protection, not a faithful DevDataBench reproduction.

Dropping query-relevant evidence remains a training-noise risk, explicitly part
of the test of whether the method transfers when facet annotation is unavailable.
No post-dropout relevance guarantee is asserted.
