# EMPIRE Cost Analysis: Two Routes

> **Honest framing (added Sprint 1):** The numbers below are *modeled at the
> assumed scale of 10 simultaneous streams × 50 products per seller*. As of
> today's build, EMPIRE runs **single-tenant: one product, one stream, one
> dashboard**. Multi-product is shipped (Sprint 1.3) but multi-stream
> isolation is roadmap (Sprint 5+).
>
> Two known math errors in the original write-up below, kept inline for
> transparency rather than silently fixed:
>
> 1. **RunPod cost.** The "~$50/mo for RunPod" figure assumed ~3 active
>    hours/day. A 5090 spot at $0.40/hr × 24h × 30d = **$288/mo if the pod
>    runs continuously.** For a 24/7 multilingual seller (the pitch's
>    headline), the real number is ~$288, not $50.
> 2. **Scale mismatch.** The cloud column models 10 streams × 50 products
>    × 24/7. The on-device column models one laptop's electricity. Apples
>    to oranges. To get a defensible number, both columns should be at the
>    same scale and the on-device side should account for the fact that
>    one laptop can't serve 10 simultaneous streams (CPU/mic/GPU bottleneck).
>
> Today's defensible single-stream number: **~$30 electricity + ~$60
> ElevenLabs (Starter) + ~$25 Bedrock + ~$288 RunPod = ~$403/mo for a
> 24/7 single-stream seller.** Still 12-30× cheaper than a human host
> team ($5K-$12K/mo) but not the headline $144 figure.
>
> The 90%+ gross-margin claim and the local-first thesis hold up; the
> specific dollars need a Sprint 2/3 rebuild against measured load.

---


## Route 1: Human Live Sellers (What You're Replacing)

### What humans cost

| Role | Pay | Source |
|------|-----|--------|
| TikTok Live Host (avg) | **$18/hr** ($37,825/yr) | ZipRecruiter Feb 2026 |
| TikTok Live Host (range) | **$14-57/hr** | Job listings 2025-2026 |
| NYC livestream host | **$30-35/hr + commission** | Job postings |
| LA in-studio host | **$20/hr** | Job postings |
| Budget host | **$12/hr + 2% commission** | Job postings |
| Creator/affiliate commission | **10-30% of sales** | TikTok Shop standard |

### What a typical small seller spends monthly

**Scenario: Selling 8 hours/day, 6 days/week**

```
Hired live host:
  $20/hr × 8 hrs × 26 days              = $4,160/month
  + Commission (2% on $50K GMV)          = $1,000/month
  Total:                                   $5,160/month

If you hire from an agency:
  Agency retainer                        = $3,000-5,000/month
  + Performance bonus (10-15% of sales)  = $5,000-7,500/month
  Total:                                   $8,000-12,500/month

If you go live yourself:
  Your time (8 hrs/day, valued at $30/hr)= $6,240/month (opportunity cost)
  + You're exhausted and can't do anything else
  + Limited to YOUR time zones, YOUR languages, YOUR energy levels
```

### What humans CAN'T do

- Work 24/7 (sleep, burnout, sick days)
- Speak 140+ languages
- Respond to every single comment (humans miss 60-80% during busy streams)
- Perfect product knowledge recall (humans forget details, improvise wrong specs)
- Run simultaneous streams on multiple platforms
- Consistently maintain energy for 8+ hours

### What humans CAN do (that AI currently struggles with)

- Generate genuine hype and infectious energy during flash sales
- Build parasocial relationships (viewers come back for the PERSON)
- Improvise humor and contextual jokes
- Read the room and pivot strategy in real-time

**EMPIRE's positioning:** AI handles the 24/7 grind (overnight, off-peak, multi-timezone, multi-language). Humans handle the high-energy moments (flash sales, product launches, community events). The AI pays for itself by covering the 16 hours/day the human ISN'T live.

---

## Route 2: Cloud AI vs Gemma 4 On-Device (Why On-Device Changes Everything)

### The livestream comment processing problem

A decent TikTok Live stream gets **30-100 comments per hour**. A popular one gets **500+/hour**.

Every comment needs to be:
1. **Read** (tokenized and processed)
2. **Classified** (question? compliment? objection? spam?)
3. **Response generated** (if it needs one)

This happens continuously, 24/7, for every stream.

### If you use Claude API for everything

**Claude Sonnet 4.6 pricing:** $3/million input tokens, $15/million output tokens

```
Per comment processed:
  Input: ~150 tokens (system prompt + product context + comment)
  Output: ~75 tokens (classification + response)

  Input cost:  150 tokens × $3/1M    = $0.00045
  Output cost: 75 tokens × $15/1M    = $0.001125
  Total per comment:                    $0.00158

  Per hour (50 comments):               $0.079
  Per day (24 hrs):                      $1.90
  Per month:                             $57

Not terrible for 50 comments/hour. But here's where it explodes:
```

**Scaling problem: multiple products × multiple streams × higher volume**

```
1 product, 1 stream, 50 comments/hr:
  $57/month — manageable

10 products, 3 simultaneous streams, 200 comments/hr total:
  Input context grows: each comment needs full product knowledge
  ~500 tokens input per comment × 200/hr × 24hrs × 30 days
  Input:  500 × 200 × 24 × 30 × $3/1M   = $648/month
  Output: 100 × 200 × 24 × 30 × $15/1M  = $2,160/month
  TOTAL:                                    $2,808/month

50 products, 10 simultaneous streams, 1000 comments/hr:
  Input context balloons: product catalog context per query
  ~1000 tokens input per comment (product matching + context)
  Input:  1000 × 1000 × 24 × 30 × $3/1M   = $2,160/month
  Output: 100 × 1000 × 24 × 30 × $15/1M   = $10,800/month
  TOTAL:                                      $12,960/month
```

**And that's JUST comment processing.** Add:
- Voice transcription (Whisper API): $0.006/minute × 60 min × 24 hrs × 30 days = **$259/month** per stream
- Sales script generation per product: small
- Product analysis per product: small

```
FULL CLOUD COST AT SCALE (10 streams, 50 products):

Comment processing (Claude):         $2,808/month
Voice transcription (Whisper):        $2,590/month (10 streams)
TTS (ElevenLabs, $0.30/1K chars):    $3,240/month (10 streams, continuous)
Avatar rendering (HeyGen):            $990/month (HeyGen Enterprise)
                                      ─────────────
TOTAL CLOUD:                          $9,628/month
```

### If you use Gemma 4 on-device + local tools

```
Comment processing (Gemma 4 on phone):     $0
  Reads comments, classifies, generates responses
  All on-device, unlimited

Voice understanding (Gemma 4 native audio): $0
  No transcription needed — direct audio understanding
  On-device, unlimited

TTS (Sesame CSM on laptop):                $0
  Open source, runs locally on GPU
  Unlimited generation

Avatar lip-sync (LiveTalking on laptop):    $0
  Open source, runs locally
  Unlimited

Complex questions routed to Claude (5-10%): $57-114/month
  Only the hard questions that Gemma 4 can't handle

Electricity for laptop running 24/7:        ~$30/month

                                            ─────────────
TOTAL ON-DEVICE:                            $87-144/month
```

### Side-by-side comparison

```
MONTHLY COST AT SCALE (10 streams, 50 products, 24/7)

┌─────────────────────────┬───────────────┬───────────────┐
│                          │ CLOUD ONLY    │ GEMMA 4 +     │
│                          │ (Claude/GPT)  │ ON-DEVICE     │
├─────────────────────────┼───────────────┼───────────────┤
│ Comment processing       │ $2,808        │ $0            │
│ Voice transcription      │ $2,590        │ $0            │
│ TTS / voice generation   │ $3,240        │ $0            │
│ Avatar rendering         │ $990          │ $0            │
│ Complex Q fallback       │ (included)    │ $114          │
│ Electricity              │ ~$0 (cloud)   │ $30           │
├─────────────────────────┼───────────────┼───────────────┤
│ TOTAL                    │ $9,628/mo     │ $144/mo       │
├─────────────────────────┼───────────────┼───────────────┤
│ SAVINGS                  │               │ 98.5%         │
│ MULTIPLIER               │               │ 67x cheaper   │
└─────────────────────────┴───────────────┴───────────────┘
```

### All three routes in one table

```
┌────────────────────┬──────────────┬─────────────┬────────────┐
│                      │ HUMAN SELLER │ CLOUD AI    │ EMPIRE     │
│                      │              │ (all Claude)│ (Gemma 4)  │
├────────────────────┼──────────────┼─────────────┼────────────┤
│ Monthly cost         │ $5,160-12,500│ $9,628      │ $144       │
│ Hours/day            │ 8            │ 24          │ 24         │
│ Languages            │ 1-2          │ 10-20       │ 140+       │
│ Simultaneous streams │ 1            │ 10+         │ 10+        │
│ Comment response     │ Misses 60%+  │ Every one   │ Every one  │
│ Product knowledge    │ Forgets      │ Perfect     │ Perfect    │
│ Energy consistency   │ Degrades     │ Perfect     │ Perfect    │
│ Cost per stream/day  │ $160-480     │ $32         │ $0.48      │
├────────────────────┼──────────────┼─────────────┼────────────┤
│ vs Human savings     │ —            │ 23-77%      │ 97-99%     │
│ vs Cloud savings     │              │ —           │ 98.5%      │
└────────────────────┴──────────────┴─────────────┴────────────┘
```

### The punchline for judges

"A human live seller costs $5,000-12,000/month for 8 hours a day in one language.

If you try to replace them with cloud AI, you just trade the human cost for API bills — $9,600/month at scale.

EMPIRE will run 24/7 in 140 languages on 10 simultaneous streams for $144/month — *at the modeled scale described in this doc*. **As of today's build, English-only, single-stream, and the realistic single-stream cost is ~$400/mo (see honest-framing note at the top).** The local-first thesis still wins: Gemma 4 + Cactus on-device handles 90-95% of comment compute for $0 marginal cost, and that's the load-bearing claim. The phone is the server. The model is the employee. The only cloud cost is the 5-10% of questions that escalate.

On-device AI isn't a nice-to-have. It's the entire business model. Without it, the unit economics don't work."

---

## Sources

- ZipRecruiter: TikTok Live Host salaries (https://www.ziprecruiter.com/Salaries/Tiktok-Live-Host-Salary)
- Claude API Pricing (https://platform.claude.com/docs/en/about-claude/pricing)
- Sonnet 4.6: $3 input / $15 output per million tokens
- Haiku 4.5: $1 input / $5 output per million tokens
- TikTok Shop referral fees: 6% standard (https://www.podbase.com/blogs/tiktok-shop-fees)
- Creator affiliate commissions: 10-30% (industry standard)
