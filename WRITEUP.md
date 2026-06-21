# When Did the Compromise Begin? Insider Threats as Behavioral Bifurcations

*A writeup of LABAD — an insider-threat pipeline that borrows the language of
nonlinear dynamics to answer the question a security analyst actually asks.*

---

## The question nobody answers

Almost every "AI for insider threat" project ends the same way: an autoencoder
over the CERT dataset, a reconstruction-error anomaly score, an ROC curve, an
AUC of 0.9-something. Done.

But sit with a SOC analyst for an afternoon and you'll hear a different
question. Not *"is this user anomalous?"* — they already suspect the account.
The question is **"when did it start?"** Because that single timestamp decides
everything downstream: how many days of activity to audit, how much data to
assume was taken, whether legal needs to be called.

LABAD is my attempt to answer *when*, not just *whether*. The trick is to stop
thinking of a user's behavior as a pile of log lines and start thinking of it
as a **dynamical system**.

## Behavior as a dynamical system

Here's the reframe. A user's daily anomaly score is a time series — the output
of some process generating their behavior. A benign employee sits in a stable
regime: low scores, small fluctuations, day after day. When they go rogue, the
*generating process itself* changes. Not one weird day — a shift to a new
regime.

In nonlinear dynamics there's a precise word for this: a **bifurcation**, the
moment a system's qualitative behavior changes, a stable state going unstable.
An insider going rogue is a behavioral bifurcation. And there's an algorithm
built to detect exactly that, online, one day at a time: **Bayesian Online
Changepoint Detection** (BOCPD).

BOCPD tracks a posterior over the *run length* — how many days since the last
changepoint. Most days the run grows by one. When the data stops looking like
the current regime, the posterior collapses toward run-length zero. That
collapse, on a user's anomaly-score series, **is** the behavioral bifurcation —
and you can read the day it happened straight off the algorithm.

## Does it work?

For *abrupt* insiders — the ones who suddenly start working at 2am, plugging in
USB drives, and uploading files (CERT scenario 1) — yes, and well:

- **~80% localized within ±7 days** of the true onset
- **median 3 days** off
- and the mean error is *negative* — it typically fires **1–2 days before** the
  labeled onset, catching the ramp-up.

The full pipeline around it is real too: an LSTM autoencoder trained only on
benign users, a Neyman–Pearson threshold that pins the false-positive rate to a
SOC's analyst budget (AUC 0.87, 100% precision@10), and locally-hosted LLM
threat reports grounded in MITRE ATT&CK so no behavioral data ever leaves the
building. It's containerized and experiment-tracked.

But honestly, the detection numbers aren't the interesting part. The
interesting part is everything that *didn't* work, and what it taught me.

## The bug that was hiding in plain sight

The textbook way to read changepoints out of BOCPD is to threshold
`P(run length = 0)`. I implemented it, ran it on a clean synthetic step
change… and it detected nothing.

The run-length posterior was collapsing perfectly — I could see it in the data.
But `P(r=0)` sat pinned at exactly `0.02` forever. Then I did the algebra:

> P(r=0) = (H·S) / ((1−H)·S + H·S) = H

`P(r=0)` is **mathematically equal to the hazard rate at every step,
regardless of the data.** It carries *zero* information. A whole class of BOCPD
tutorials reads changepoints off a quantity that can't possibly contain them.
The real signal is the mass collapsing onto *short* run lengths — `P(r ≤ w)` —
which is what LABAD actually uses.

That's the kind of thing you only catch if you implement it yourself and refuse
to trust a number that looks wrong.

## "You can't detect what the score doesn't encode"

The gradual insiders (scenario 2 — slowly browsing job sites, then stealing
data on the way out) were a different story. BOCPD missed them. So did CUSUM. So
did everything I threw at it.

The instinct is to blame the detector and try a fancier one. I did, and it kept
failing — every variant false-alarmed immediately. So I went looking for *why*,
and found two layers:

1. **The signal is autocorrelated.** Overlapping 30-day windows make
   consecutive scores 0.83–0.99 correlated — nearly a random walk. CUSUM and
   friends assume near-independence; they never had a chance.
2. **The scalar score destroys the signal.** It averages reconstruction error
   over all 18 features. At a gradual onset the *overall* score moves
   `1.03×` — a non-event. But a *single feature* (`exe_access_count`) spikes
   **151×**. Averaging 17 calm features with one screaming one erases it.

So I scored per-feature instead. It recovered the sharp insiders *and* told me
which behavior changed (explainability the scalar can't give). But scenario-2
onset *still* wouldn't localize — and the final diagnostic is my favorite
result in the whole project:

> For scenario-2 users, the suspicious behaviors are **more active *before* the
> labeled onset than after** (job-site visits: 22.5% of pre-onset days vs 14.3%
> after).

The CERT "onset" label marks the final exfiltration act — *not* a behavioral
discontinuity. There is **no bifurcation at the label to find.** A bifurcation
detector that reports "nothing changed here" is being *correct*. The failure was
in the data, not the method — and proving that, three independent ways, is worth
more than a detection number would have been.

## The physics made a prediction — and it held

Here's where the dynamical-systems framing stopped being a metaphor and became
falsifiable. Tipping-point theory says a system approaching a bifurcation loses
resilience, which shows up as **rising variance and autocorrelation
*before* the transition** — "critical slowing down," the same early-warning
signal used for ecosystem collapse and financial crashes.

If an insider's onset is a real bifurcation, those signatures should appear in
the weeks *before* it. So I tested it. They do:

> Critical-slowing-down warnings are **2.28× enriched** in the 45 days before an
> abrupt onset (24/30 users; binomial *p* = 7×10⁻⁴, Wilcoxon *p* = 2×10⁻⁴),
> robust across parameters.

It's not a standalone alarm yet — the baseline rate is too high — but it's a
statistically significant *predictive* signature. The physics framing made a
prediction about data it had never seen, and the data agreed.

## Why I think this matters beyond CERT

The reusable idea underneath LABAD is simple: **detect when an entity's behavior
changes regime, on a stream of its actions.** That's not specific to human
insiders.

As organizations deploy autonomous LLM agents with real tool access, the next
insider is a *compromised, prompt-injected, or goal-drifted agent*. Its
telemetry is the same kind of behavioral time series. The question — "when did
this agent's behavior bifurcate?" — is the same question, and the same
machinery applies.

That's the thread I find worth pulling: insider-threat detection as a special
case of behavioral-bifurcation detection, and bifurcation detection as
something the AI-agent security era is going to need.

---

*Code, the full technical report, and reproduction instructions:
[github.com/Y-ash-Y/LABAD](https://github.com/Y-ash-Y/LABAD).*
