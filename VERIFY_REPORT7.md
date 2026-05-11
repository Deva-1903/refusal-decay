# Verify Report 7

## COMPLETE

- Behavioral: harmful_k00: 25 rows, 0 errors
- Behavioral: harmful_k03: 25 rows, 0 errors
- Behavioral: benign_k00: 25 rows, 0 errors
- Tracing: Tracing summary covers layers 16/20/24/27 and harmful_k00/harmful_k03.
- Prompt-Level Association: Prompt association includes layers 24/27 and both label groups.
- Cross-Condition Patching: Cross-condition patching valid rows=300, layers=[16, 24, 27], targets=[0, 1, 3, 5]
- Additive Intervention: Additive intervention valid rows=4950, alphas=[1.0, 2.0], directions=['orthogonal', 'random', 'refusal']

## MISSING

- None

## SUSPICIOUS

- None

## READY TO WRITE REPORT 7?

Yes

## RUN THESE NEXT

- Report 7 outputs are ready for writeup.
