# Sentinel governance corpus

Hand-authored control catalogs consumed by the ingest pipeline (`POST /ingest`).

## File contract
Each file is a YAML mapping `{framework, controls}`: `framework` is the
framework label the loader stamps on every row (`ISO` or `CIS`), and `controls`
is a list of entries. Every entry has EXACTLY three keys:

- `control_id` — canonical id. ISO: `A.<clause>.<num>` (e.g. `A.5.1`).
  CIS: `CIS-<n>` for a control, `CIS-<n>.<m>` for a safeguard.
- `title` — short human title.
- `summary` — 1-4 sentence ORIGINAL paraphrase of the control's intent.

## Authoring rules (MANDATORY)
1. Original wording only. Do NOT copy verbatim ISO/IEC 27001:2022 text
   (copyrighted) or CIS text. Paraphrase the intent in your own words.
   CIS Controls v8 material is used under CC BY-NC-SA; ISO is summarized, not reproduced.
2. Exactly three keys per entry. No extra keys, no missing keys.
3. Unique `control_id` within a file.
4. UTF-8. All reads/writes use `encoding="utf-8"`.
5. Keep entries ordered by clause/number for reviewability.

## Expected counts (enforced by @pytest.mark.corpus tests)
- ISO: 93 controls across A.5 (37) / A.6 (8) / A.7 (14) / A.8 (34).
- CIS: exactly 18 `CIS-<n>` controls + 153 `CIS-<n>.<m>` safeguards.

## Attribution / licensing
- CIS: source is the CIS Critical Security Controls v8 by the Center for Internet
  Security, Inc., licensed CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/). The summaries in
  `cis_v8.yaml` are adapted (paraphrased) from it, and that file is distributed
  under CC BY-NC-SA 4.0.
- ISO: the summaries in `iso27001.yaml` are original paraphrases, not ISO text.
  ISO/IEC 27001:2022 is copyright ISO/IEC.
- Per-source provenance and license details: see [SOURCES.md](SOURCES.md).
