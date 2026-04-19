# Rehearsal Phrases — whisper validation for stage

Whisper-base on the iPhone mic is ~4/5 reliable on clean speech. For stage
we want ≥9/10 per phrase — anything less is a 20%+ live-fail risk at the
worst possible moment.

Each candidate phrase below is mapped to a specific `qa_index` keyword in
[backend/data/products.json](../backend/data/products.json). A rep passes
only if **all three** happen:

1. Whisper transcribes the right keywords (see "Transcript must contain")
2. Swift `Router.decide()` dispatches `respond_locally` with the expected `answer_id`
3. The correct MP4 plays (router-targeted `/local_answers/wallet_*.mp4`)

Anything else = fail. Count strictly.

**Ship rule:** need at least **5 phrases at ≥9/10** to walk on stage.
Discard phrases that don't clear; replace from the backup bank below if
you drop below 5.

---

## Candidate bank

| # | Phrase | Maps to | Transcript must contain | Expected MP4 | 10-trial scorecard | Score | Ship? |
|---|---|---|---|---|---|---|---|
| 1 | "How much does it cost?" | `price` | "cost" OR "how much" | `wallet_price.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 2 | "Is this real leather?" | `is_it_real_leather` | "real leather" OR "leather" | `wallet_real_leather.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 3 | "Does it ship internationally?" | `how_does_it_ship` | "ship" OR "shipping" | `wallet_shipping.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 4 | "What's the return policy?" | `return_policy` | "returns" OR "return policy" | `wallet_returns.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 5 | "How long is the warranty?" | `warranty` | "warranty" OR "guarantee" | `wallet_warranty.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 6 | "What colors does it come in?" | `color` | "color" OR "colors" | `wallet_colors.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |
| 7 | "How many cards does it hold?" | `card_capacity` | "cards" OR "hold" | `wallet_cards.mp4` | ☐☐☐☐☐☐☐☐☐☐ | /10 | — |

## Backup bank (swap in if main loses a phrase)

| # | Phrase | Maps to | Transcript must contain | Expected MP4 |
|---|---|---|---|---|
| B1 | "Is it waterproof?" | `water_resistance` | "water" OR "waterproof" | `wallet_water.mp4` |
| B2 | "Does it have RFID blocking?" | `rfid` | "rfid" OR "blocking" | `wallet_rfid.mp4` |
| B3 | "How big is it?" | `sizing` | "big" OR "size" | `wallet_sizing.mp4` |

---

## How to run a rep

1. Backend + dashboard up (`make start`); backend has wallet as active product
2. iPhone on same network as Mac; long-press status pill → verify host IP
3. Open EmpirePhone, wait for "READY"
4. Hold button, speak phrase at natural pace (don't over-enunciate)
5. Release → WHISPER card should show transcript within ~350ms
6. Decision card (ROUTER) should show `respond_locally` within ~1ms
7. MP4 plays on the phone; GEMMA card lights magenta 2-3s later
8. Mark the rep:
   - **Pass** = transcript has the keyword + ROUTER = respond_locally + right MP4 played
   - **Fail** = anything else (wrong transcript / cloud escalate / wrong MP4 / silence)

Track reps in the scorecard column above. Cross off (☒) each one. After 10 reps, total → Score → Ship? (Y if ≥9).

## Transcript history

After P0.1 shipped, the iPhone shows the last 5 transcripts above the active
one. Use that to double-check what whisper heard over the last reps — spot
patterns (always misses "ship" on specific venues' audio, etc.) without
re-reading logs.

Tap the "HISTORY · tap to clear" header to reset between phrases.

## Known whisper pitfalls

- **Soft consonants drop:** "ship" can become "sip" or "ssh" in noisy audio. Mitigation: prefer phrases with harder consonants if a soft one is failing.
- **Filler words add noise:** "um, how much does it cost" often trips. Mitigation: clean starts/stops. Press → speak → release.
- **Rising pitch at end:** some phrases work better as statements than questions. "It costs how much" sometimes beats "How much does it cost?"
- **Ambient noise floor:** Venue matters. Run the full 10 reps in the actual demo environment if possible — NOT in your kitchen.

## If you can't get 5 to ≥9/10

Options (in order of escalation):
1. Try the 3 backup phrases
2. Swap the failing phrase for a simpler variant ("cost?" vs "how much does it cost?")
3. Add a **pinned** cuing behavior: lower phone volume on stage so whisper isn't fighting the speaker's echo
4. Accept the risk and prepare backup patter: "Sometimes whisper fumbles — let me try again" (honest, humanizing)
5. Skip straight to the backup video

Never ship a phrase that scored <9/10. The cost of a live fail on stage is higher than the cost of using fewer phrases.
