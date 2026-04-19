# EMPIRE — Hackathon Pitch Brief

## Part 1: What Wins This Hackathon

This is the Gemma 4 Voice Agents Hackathon at YC, hosted by Cactus (YC S25) + Google DeepMind. April 18-19. ~200 builders from CMU, Harvard, Yale, top schools across the country. One winner gets a guaranteed YC interview.

### What judges care about (reverse-engineered from past YC hackathon winners)

1. **Startup potential.** This is YC. They're not judging a science fair. They want to see something that could be a company. The first question in their heads: "Would I fund this?"

2. **Live demo, not slides.** Every single YC hackathon winner had a working demo. Riley Shu (won both YC and OpenAI hackathons) said: "Avoid slides. Focus on live demo. Design for wow effect." The demo IS the pitch.

3. **Technical depth.** Against 200 elite builders, a simple chatbot loses. MedSim won "Most Innovative" at Harvard HSIL by combining World Labs 3D world models + an agent swarm + real-time simulation. The winning project combines frontier technologies in a way nobody has combined them before.

4. **"Useful + unexpected."** Not just clever tech. Not just a real problem. The intersection. The thing that makes judges say "I didn't know that was possible AND I want this to exist."

5. **Clear 2-sentence pitch.** YC partners ask every team: "How would people use this?" If you can't answer that crisply in two sentences, you lose.

6. **3 features that work perfectly > a platform where nothing works.** Scope tight. Polish the demo path. Don't show broken edges.

### What the hackathon specifically requires
- Must use Gemma 4 + Cactus (the sponsors' tech)
- Voice as the primary interface
- Working demonstration
- Architecture explanation with design trade-offs

---

## Part 2: The Market — Why Live Commerce Is The Opportunity

### What is live commerce?

Live commerce = selling products through livestreams. A host shows a product, talks about it, answers questions from viewers in real-time, and viewers buy directly from the stream.

Think QVC/Home Shopping Network but on TikTok, Instagram, and YouTube, run by individual sellers, not TV networks.

### The numbers

| Metric | Number | Source |
|--------|--------|--------|
| US live commerce market (2026) | **$68 billion** | TikTok Stats |
| Global live commerce (China) | **$500 billion** | McKinsey |
| TikTok Shop global sales (last quarter) | **$19 billion** | Industry reports |
| US TikTok Shop growth (YoY) | **+125%** | TikTok |
| Live commerce conversion rate | **30%** | Industry average |
| Traditional e-commerce conversion | **2-3%** | Industry average |

Live commerce converts at **10x** the rate of traditional e-commerce. That's not a marginal improvement. That's a different category.

### Why live commerce is eating traditional e-commerce

- **Trust.** You see the product in someone's hands. You see the texture, the size, how it moves. Photos lie. Live video doesn't.
- **Urgency.** "Only 12 left at this price" works in real-time. It doesn't work on a static product page.
- **Interaction.** "Does it come in blue?" — answered in 3 seconds. On a product page, you'd have to email customer service and wait 24 hours.
- **Discovery.** People don't go to TikTok to shop. They discover products while being entertained. The content IS the store.

### The problem: going live is HARD

Here's why most sellers don't do it, even though the numbers are insane:

1. **You need to be on camera.** Most people hate being on camera. It's terrifying.
2. **You need to be entertaining AND informative AND responsive simultaneously.** That's a professional skill most people don't have.
3. **You need equipment.** Good lighting, good audio, multiple camera angles for professional quality.
4. **You need a team.** Professional livestream sellers have: a camera operator, a producer managing comments, a graphics person for overlays, a sales coach. That costs $1,000-10,000 per stream.
5. **You can only be live for a few hours a day.** Sleep exists. But your potential customers are in every time zone.

**The result:** The sellers who CAN go live make insane money. The 95% who can't are locked out of a $68B market.

### What's already happening with AI in this space

- TikTok launched AI Seller Assistant, automated live highlights, and chat auto-response
- Khaby Lame sold his AI avatar rights for $975 million (avatar can sell in his likeness across languages)
- HeyGen launched LiveAvatar — real-time interactive AI avatars via API
- Top streamers already use AI overlays for comment management and product tagging
- The entire industry is moving toward AI-augmented selling. First mover advantage is NOW.

---

## Part 2.5: Early Validation

EMPIRE isn't a mockup. The hackathon build is a working alpha. Observed numbers from this weekend's internal tests (Apr 18-19, 2026):

| Metric | Observed | Notes |
|---|---|---|
| Voice → avatar speaking (local path) | **under 600 ms** | Cactus whisper 244 ms + rule-based router 0 ms + pre-rendered MP4 |
| Voice → avatar speaking (cloud escalate) | **4.5 – 5.5 s warm** | Bedrock Claude + ElevenLabs + Wav2Lip on RTX 5090 |
| Comments handled without cloud round-trip | **~90%** | 10 product Q&A entries + compliments + spam all stay local |
| Cost per escalated comment | **$0.00035** | Claude Haiku 1K tokens input / 150 tokens output |
| Cost per *avoided* cloud call | **$0** | Pre-rendered MP4 served from disk |
| Monthly spend for a 24/7 avatar (modeled) | **$144** | vs $9,628/mo for a live human host team |

These numbers are from our demo product — a minimalist leather wallet with a 10-entry Q&A index. A real seller onboards in ~15 minutes (product photo → Gemma 4 vision analyses it → author 10–20 answers → pre-render).

**What this proves:** on-device routing isn't a tech-flex. It's the thing that makes 24/7 AI live commerce economically possible. Without it you're paying $9K/mo in cloud LLM tokens to talk to your own customers. With it you're paying electricity.

---

## Part 3: The Idea — EMPIRE

### One-liner

**"Put a product on a table. Say 'sell this.' An AI agent swarm builds your entire commerce operation — product photos, 3D model, virtual showroom, AI salesperson that goes live, customer service — all from a phone."**

### What it does

You put ANY product on a table. You say: **"Sell this for $49. Target young professionals."**

An agent swarm activates. Five specialized AI agents, coordinated by voice:

---

**AGENT 1: EYES (Product Intelligence)**
*Runs on: Gemma 4 on Cactus (on-device)*

Sees the product through the phone camera. Identifies everything: what it is, materials, dimensions, quality markers, unique selling points, comparable products. Generates a product knowledge base that ALL other agents reference.

Why on-device: Your product might be pre-launch. Sending photos to a cloud server before launch = IP risk. The product knowledge stays on your phone until YOU decide to go live.

---

**AGENT 2: CREATOR (Content Factory)**
*Runs on: Gemma 4 vision (on-device) + TripoSR (3D gen) + World Labs Marble API (3D world)*

Takes the product knowledge from EYES + camera frames. Generates:
- 7 marketplace-ready product photos (multi-angle, white background, lifestyle context)
- A 3D model of the product (TripoSR: single image to 3D mesh in 0.5 seconds)
- A virtual showroom environment (World Labs: text to navigable 3D world in ~5 min)
- Places the 3D product inside the 3D showroom (customers can explore)
- A 30-second promo video with cuts, text overlays, and pacing
- 3-5 short-form clips formatted for TikTok, Reels, Shorts

---

**AGENT 3: SELLER (Live Commerce)**
*Runs on: Gemma 4 on-device (brain) + HeyGen LiveAvatar or NVIDIA Audio2Face (face) + Sesame CSM (voice)*

An AI avatar that goes live on TikTok/Instagram. It:
- Knows everything EYES identified about the product
- Talks naturally with human-quality voice (Sesame CSM — ums, pauses, emphasis)
- Demonstrates features it saw through the camera ("Look at the stitching detail here")
- Responds to live comments in real-time (Gemma 4 on-device reads comments, generates responses, avatar speaks them)
- Handles objections ("Is it real leather?" — "Yes, full-grain leather. You can see the natural texture variation.")
- Creates urgency, drives conversions
- Runs 24/7. You sleep. SELLER sells.

---

**AGENT 4: CLOSER (Customer Intelligence)**
*Runs on: Gemma 4 on-device*

Handles all inbound DMs and messages across platforms. "When does it ship?" "Do you have it in blue?" "Can I return it?" — all answered instantly, in the seller's tone, in the buyer's language (140+). Escalates to the human only for genuine edge cases.

---

**AGENT 5: BRAIN (Optimization Loop)**
*Runs on: Gemma 4 on-device*

Watches everything. Tracks what converts. Learns.
- "Lifestyle photos convert 3x better than white background for this category"
- "Mentioning the stitching detail increased add-to-cart by 40%"
- "Comments about sizing come up in 60% of streams — address it proactively"

Feeds improvements back to CREATOR and SELLER in real-time. The system gets smarter every hour it runs.

---

### The tech stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Product understanding | Gemma 4 E4B on Cactus | See the product, understand it, generate knowledge base |
| Voice interface | Gemma 4 native audio on Cactus | Understand voice commands, direct the swarm |
| 3D product model | TripoSR | Single photo to 3D mesh in 0.5s |
| Virtual showroom | World Labs Marble API | Text/image to navigable 3D gaussian splat world |
| AI avatar face | HeyGen LiveAvatar API or NVIDIA Audio2Face-3D | Real-time photorealistic talking avatar |
| AI avatar voice | Sesame CSM (open source, 1B params) | Human-quality speech with emotion and natural disfluencies |
| Agent coordination | CrewAI or custom swarm | Multi-agent orchestration |
| Content generation | On-device Gemma 4 + cloud image gen | Product photos, videos, clips |
| Comment handling | Gemma 4 on Cactus | Real-time understanding and response generation |

### Why on-device (Gemma 4 + Cactus) is core, not cosmetic

1. **Pre-launch IP protection.** Products photographed and analyzed on-device never touch a server until you go live. For sellers with unreleased products, this is non-negotiable.
2. **Selling strategy is proprietary.** The BRAIN agent's knowledge of what converts is your competitive advantage. Can't live on HeyGen's servers.
3. **Real-time comment response.** Today: 2-4s classify on CPU prefill (target <500ms with the upcoming Cactus NPU mlpackage). Cloud adds another 1-2s on top of model time — on-device still wins on latency for routine comment classes, and the local-first router lets us play a pre-rendered MP4 in <1s for the 90% of comments that match the Q/A index.
4. **Customer conversations are private.** DMs contain personal info, addresses, payment discussions.
5. **Zero marginal cost.** Processing 1,000 comments per stream on cloud APIs = expensive. On-device = free.

### Why this is a multi-billion dollar company

- **TAM:** 200M+ small businesses globally need to sell online. 10M+ active TikTok Shop sellers. $68B US live commerce market growing 100%+ YoY.
- **Revenue model (three tiers):**
  - **Solo sellers — $99/mo.** One product, unlimited streams. Target the 10M TikTok Shop creators already selling.
  - **Brands — $499/mo.** Multi-product catalog, BRAIN analytics, outbound DM campaigns.
  - **Agencies & fulfillment houses — $5K–$50K/mo.** The real enterprise wedge. Agencies managing 20–200 brands today pay for human live-host teams ($10K–$40K per brand per month). EMPIRE replaces one agency's entire talent roster with one Mac. Agencies have procurement budgets and 3-year contracts — this is the fastest path to $10M ARR.
  - Zero server cost per seller on the local path = **90%+ gross margins.**
- **Moat + defensibility window:** Every stream teaches the BRAIN agent which questions convert. After 100 sellers × 10,000 streams, category-level conversion patterns become proprietary ("lifestyle photos convert 3× on leather goods; address sizing proactively in 60% of apparel streams"). **First-mover window: 6 months** before HeyGen or Shopify ship an equivalent. After that, accumulated conversion data + on-device weights routing are the defensible layer — the code is table-stakes. Network effects accelerate: more sellers → more conversion data → better AI → stickier sellers.
- **Growth:** Every livestream has a watermark. Sellers are public by nature. Viral distribution built into the product.
- **Timing:** Live commerce is exploding NOW. TikTok Shop US grew 125% last quarter. Khaby Lame's AI avatar sold for $975M. The market has validated that AI selling is worth nearly a billion dollars. But no one has built the full-stack AI commerce engine. First mover wins.
- **Comp:** HeyGen does avatars but not commerce intelligence. TikTok's AI tools are features, not a platform. Shopify gives you a store but no sales team. Empire gives you everything.

### The 2-minute demo

**[0:00-0:15]** Place a product on the table. "Build me a brand. Sell this for $89. Target women 25-35." Show the agent swarm activating — five agents light up on the dashboard.

**[0:15-0:40]** EYES analyzes the product. Show Gemma 4's output on screen: "Leather crossbody bag, brass hardware, hand-stitched seams, approximately 10x7 inches." CREATOR starts generating product photos — they appear one by one. Show the 3D model spinning.

**[0:40-1:10]** The AI SELLER avatar appears. Starts presenting naturally: "This hand-stitched leather crossbody is the kind of thing you buy once and carry for ten years. Look at this brass hardware — that's not plated, that's solid brass." The avatar references features that EYES actually identified from the camera. It's specific to THIS product.

**[1:10-1:40]** A judge types a comment: "Is it real leather?" The SELLER pauses, reads the comment, responds: "Great question — yes, this is full-grain leather. See the natural grain variation? That's how you know it's real." Another comment: "What colors?" CLOSER agent responds in DMs simultaneously. Show both happening at once.

**[1:40-2:00]** Show the full output: 7 product photos, 3D model, promo video, live stream running, 3 TikTok clips auto-generated, DMs being handled. "Five minutes ago, this was a bag on a table. Now it's a business. This is what AI-native commerce looks like."

### Team, Ask, Milestones

> Team and ask blocks are placeholders — Cody, fill in names/background/numbers before submission.

**Team:**
- **Cody Kandarian** — Founder / full-stack. Built EMPIRE end-to-end at the hackathon (Cactus SDK + Gemma 4 integration, FastAPI backend, React dashboard, pod orchestration, pre-render pipeline). [Add: prior startup / company / shipping experience.]
- **[Teammate name]** — [Role]. [Background: school, prior shipping experience, what they owned in the build.]

**Raising $1.2M seed** to close 50 seller design partners by Aug 2026 and ship the self-serve product. Use of funds: 60% engineering (two more engineers to own AGENT-3 SELLER avatar pipeline + BRAIN analytics), 25% GTM / founder-led sales into live-commerce agencies, 15% compute + pod infrastructure.

**18-month milestones:**
- **Aug 2026** — 50 seller design partners running weekly streams. Measurable conversion lift vs no-avatar control.
- **Dec 2026** — $500K ARR across SMB + first 3 agency pilots.
- **Feb 2027** — First enterprise agency contract ($15K+/mo). Series A conversations.
- **Aug 2027** — 500 sellers, $3M ARR, category-level BRAIN insights defensible.

### The 2-sentence pitch

"There are 200 million small businesses that can't compete online because they can't afford a photo studio, a film crew, a salesperson, and a customer service team. Empire is an AI agent swarm that does all of it from a phone — point at a product, say 'sell this,' and a full commerce operation spins up in minutes."
