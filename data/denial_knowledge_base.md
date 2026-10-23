# Denial resolution knowledge base (project-written demo guidance)

> Written for this portfolio prototype. These are general revenue-cycle
> practices, NOT payer policy and NOT legal or coding advice. Each section is
> one retrievable passage. Official code meanings come only from
> `carc_rarc_reference.csv`.

## KB-CORRECTED-VS-APPEAL: Corrected claim or appeal?
A denial for missing or invalid data (CARC 16 with remark codes such as N264,
N265, N290, M76, M77, or MA130) is usually fixed by sending a corrected or new
claim with the right data, not by writing an appeal. MA130 explicitly says the
claim is unprocessable and has no appeal rights. Appeals are for disputes about
a payer decision, for example medical necessity (CARC 50) or timely filing
(CARC 29) when proof of timely submission exists.

## KB-ORDERING-PROVIDER: Ordering or referring provider missing
Diagnostic tests and imaging are often ordered by a different clinician than the
one billing. When the remittance cites N264 (ordering provider name) or N265
(ordering provider identifier), locate the original order or requisition, record
the ordering clinician's name and NPI exactly as enrolled, and send a corrected
claim. Do not guess the ordering provider; if the order cannot be found, ask the
practice.

## KB-RENDERING-NPI: Rendering provider identifier missing
N290 means the rendering provider's identifier is missing or invalid. Confirm
which clinician performed the service, look up their NPI in the NPPES registry,
and confirm it passes the NPI check digit before resubmitting.

## KB-BILLING-NPI: Billing provider identifier missing
N257 refers to the billing provider/supplier identifier. Verify the group or
organization NPI and that it matches the payer enrollment record.

## KB-PRIOR-AUTH: Missing prior authorization
CARC 197 means precertification or authorization was absent. Check the
practice's authorization log for an approval number. If one exists, add it and
resubmit. If none exists, check whether the payer allows retro-authorization;
otherwise a human must decide whether to appeal. Never fabricate an
authorization number.

## KB-MEDICAL-NECESSITY: Medical necessity denials
CARC 50 means the payer did not consider the service medically necessary based
on what it received. Review the payer's published coverage policy (for Medicare,
NCDs/LCDs in the CMS Medicare Coverage Database) and the clinical notes. An
appeal letter should cite the specific documentation that exists in the record.
Only a clinician or authorized staff can attest to clinical facts. The tool must
leave placeholders for evidence rather than inventing it.

## KB-TIMELY-FILING: Timely filing denials
CARC 29 means the filing limit expired. Appeals are generally only successful
with proof of timely original submission, such as a clearinghouse acceptance
report dated within the limit. Without that proof, the balance is usually
written off per practice policy.

## KB-ELIGIBILITY: Patient not identified or coverage ended
CARC 31 (patient cannot be identified) and CARC 27 (coverage terminated) are
eligibility problems. Run an eligibility check for the date of service, compare
the member ID, name, and date of birth exactly as on the card, and correct or
redirect the claim to the correct coverage.

## KB-DIAGNOSIS-MISMATCH: Diagnosis inconsistent with procedure
CARC 11 means the diagnosis is inconsistent with the procedure. First check the
diagnosis pointers on each line: a pointer to the wrong diagnosis is a common
data-entry error that a corrected claim can fix. If the coding was correct,
review documentation before considering appeal.

## KB-BUNDLING: Bundled or same-day conflicts
CARC 97, CARC 236, and RARC N20 indicate that the payer bundled or rejected a
service billed with another same-day service. Modifiers such as 25 or 59 may be
appropriate only when documentation supports a separate, distinct service.
Adding a modifier only to bypass an edit is improper.

## KB-COB: Coordination of benefits
CARC 22 and RARC MA04 or N4 relate to other insurance. Confirm primary and
secondary coverage with the patient, bill the primary payer first, and attach
the primary payer's remittance/EOB to the secondary claim.

## KB-DUPLICATE: Duplicate claims
CARC 18 indicates an exact duplicate. Check claim history before resubmitting.
If two identical services were truly performed, a corrected claim with an
appropriate repeat-service modifier and documentation may be needed.

## KB-APPEAL-LETTER: What a good appeal letter contains
Claim number, patient identifiers as on the claim, date of service, codes
billed, the denial codes, a short factual statement of why the decision should
be reconsidered, and a list of enclosed documentation. Every factual statement
must be supported by the medical record. Drafts produced by this tool require
human review and are never submitted automatically.
