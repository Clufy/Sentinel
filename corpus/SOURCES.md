# Corpus sources and licenses

Provenance and license of each framework in the corpus. The authoring rules and
a short attribution summary live in [README.md](README.md); this file is the
per-source record.

## NIST SP 800-53 rev5

- **Source:** NIST OSCAL catalog (JSON), loaded at ingest time from
  `https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json`
  (see `app/ingest/load_nist.py`). It is not stored in `corpus/`.
- **License:** US government work, **public domain** (17 U.S.C. 105).
- **Citation form:** `[NIST AC-2]`.

## ISO/IEC 27001:2022 Annex A

- **File:** `corpus/iso27001.yaml`.
- **Text:** IDs and titles are factual references. The summaries are original
  paraphrases, not ISO text; no verbatim ISO text is stored.
- **License:** ISO/IEC 27001:2022 is copyright ISO/IEC. Our summaries do not
  reproduce it.
- **Citation form:** `[ISO A.5.1]`.

## CIS Critical Security Controls v8

- **File:** `corpus/cis_v8.yaml`.
- **Source:** CIS Critical Security Controls v8 by the
  Center for Internet Security, Inc., licensed under CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Changes made:** the summaries are adapted (paraphrased) from the source
  work.
- **License of this file:** `corpus/cis_v8.yaml` is distributed under
  CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).
  In practice: `corpus/cis_v8.yaml` may not be used for commercial purposes,
  and anything derived from it must be shared under the same CC BY-NC-SA 4.0
  licence.
- **Up-to-date guidance:** this file is a v8 snapshot. Refer to
  https://www.cisecurity.org/controls/ for the most up-to-date CIS Controls guidance.
- **Citation form:** `[CIS CIS-5]`.

## Cross-framework mappings

`corpus/mappings.yaml` (the crosswalk and `by_control` links) is original work
of this project.

## Test fixtures

`tests/fixtures/mini_nist.yaml` and `mini_iso.yaml` are small original paraphrase
fixtures used for CI. `tests/fixtures/mini_cis.yaml` is adapted from CIS Critical
Security Controls v8 (Center for Internet Security, Inc.) under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/) and carries the same licence
terms as `corpus/cis_v8.yaml`.
