# Representation and fitting audit

59,983 distinct product-setting swaps audited; zero failures. Every selected target fits C0 and all enumerated variants. The pair-swap count is larger when the same product is relevant to multiple sampled queries.

The frozen Parquet audit stores original/variant SHA-256 hashes, native input hashes, token lengths and limits, 1-based swap-position lists, canonical hashes and each individual assertion. Attribute-entry Counter equality retains duplicates; partitioned key/value tuples and numeric substring Counters are unchanged. Fixed serializer blocks are compared directly. All non-swapped positions agree exactly. Identity serialization and the existing repository audit pass. Ascending canonical equality uses (casefold(entry), entry) and is structural.

The eligibility inventory includes no-op and duplicate-serialization counts and explicit exclusions. Identical output strings are deduplicated while their original positions are retained. Targets already failing C0 are rejected without tokenizing their variants; all retained targets have a complete variant scan. No new retrieval outcomes were used for eligibility or selection.
