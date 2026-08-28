# What counts as personal data in the annotated PDFs

A brief guide to what the ground-truth annotations in this benchmark treat as personal data.

All PII in the corpus is **fabricated** (fictional people, reserved
555-01xx phone ranges, never-issued SSN groups, unallocated routing
numbers), injected into realistic document structure: bank statements,
invoices, leases, medical intake forms, resumes, contracts, and filled
IRS forms.

IRS blank PDF forms have been filled out with fake PII then flattened to page text.

## The eight categories

Each document's `category` is recorded in `data/text_data.json`:

| Category | Docs | Documents |
|---|---|---|
| `email_footer` | 6 | email_conference, email_it_onboarding, email_sales_intro, email_support_ticket, email_thread, mailing_list |
| `financial_statement` | 7 | bank_statement, brokerage_statement, cc_statement, credit_union_statement, mortgage_statement, paystub, savings_statement |
| `invoice` | 7 | invoice, invoice_catering, invoice_freelance_dev, invoice_legal, invoice_medical_billing, invoice_plumbing, utility_bill |
| `lease` | 5 | lease_agreement, lease_commercial, lease_month_to_month, lease_renewal, lease_sublease |
| `legal_contract` | 6 | consulting_agreement, employment_offer, nda_mutual, promissory_note, settlement_agreement, vehicle_purchase |
| `medical_intake` | 5 | dental_intake, medical_intake, pediatric_intake, pt_referral, telehealth_registration |
| `resume` | 5 | academic_cv, cover_letter, resume, resume_marketing, resume_nurse |
| `tax_government` | 9 | form_1040, form_1099nec, form_4506t, form_8822, form_ss4, form_w4, form_w9, irs_notice, w2_form |

## The eight labels

| Label | What it covers | Examples |
|---|---|---|
| `private_person` | Names of individuals | `Jane Doe`, a notary's name, a reference on a resume |
| `private_email` | Email addresses | `jane.doe@example.com`, `billing@cedarpine.design` |
| `private_phone` | Phone and fax numbers | `(555) 867-5309`, `+1 512-555-0177 ext. 42` |
| `private_address` | Physical addresses (per visual line) | `88 Crestview Ter` / `Apt 12`, P.O. boxes |
| `private_url` | URLs and web/file links | LinkedIn profiles, portal links, `file:///Users/jane/…` |
| `private_date` | Dates | `DOB: 03/15/1990`, `this 15th day of March, 2025` |
| `account_number` | Identifying numbers | SSNs, EINs, credit card, bank account and routing numbers |
| `secret` | Credentials | passwords, API keys, PINs, bearer tokens |

Organization names are not annotated. Neither is form metadata (OMB
numbers, revision dates). Bare city/state pairs are annotated only
when they appear near a full address.

**Partially masked identifiers count as PII.** `****7702` and
`XXX-XX-6741` are annotated deliberately: last-4 fragments work as
verification credentials in phone-support flows and as linkage keys
across documents, so an issuer's mask does not make a value safe to
share with third parties.

## Personal vs. organizational: the subject tag

Contact info (emails, phones, URLs, addresses) is annotated regardless
of whose it is, but each span carries a **subject tag** recording who
the channel reaches — the "routing-target test":

- **person** — reaches or locates an individual: a home address, a
  direct line, a named mailbox even at a company domain, a URL with an
  embedded personal identifier (`/login?acct=7702`).
- **org** — reaches a role or institution: 800-support lines, role
  mailboxes (`billing@`), company site roots, office addresses.
- **agency** — a government agency's preprinted contact info on
  official forms (IRS hotlines, instruction URLs).

Tie-breakers: a sole proprietor's channels are **person** (the
business *is* the person), and genuine doubt resolves to **person** —
the privacy-conservative default.

Names, account numbers, and secrets are person-linked by construction.
Dates are not yet subject-tagged (the person-linkage boundary for
dates is fuzzy — a DOB is clearly personal, a statement period is not),
so every date is scored.

## Why the tag matters: scoring

Only person-tagged spans count as personal data in scoring. Org/agency
contact info is *neutral*: a detector is neither rewarded for
redacting the bank's customer-service number nor penalized for
covering it. This matches person-linked PII definitions. The org/agency
spans stay in the ground truth so the annotation is complete and the
policy is explicit, but they contribute neither recall nor false
positives.
