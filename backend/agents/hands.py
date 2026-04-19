"""Hands — actuation agent.

Per EMPIRE PDF: "Hands — actuation. Posts to TikTok Shop / Shopify / Etsy.
Drops UGC clips to the FYP feed when the seller isn't live. Calls the
warehouse to ship. Job: close the loop from 'AI is talking' to 'money in
the seller's bank account.'"

v1 (this file) scope: skeleton + 4 adapters, all mocks for now. Each
adapter has the same interface — `.publish(product) -> PublishedListing`
and `.health() -> HealthStatus` — so real implementations can swap in
one-at-a-time without touching the Hands orchestrator.

Real Shopify Storefront API integration is a follow-up (requires dev-store
credentials + Storefront API token). Real TikTok Shop requires partner
approval per the PDF. Etsy has no live-commerce API (stays mock forever).
Instagram Live requires Meta Graph API partner approval.

Broadcasts:
  hands_published   — a platform just accepted a product listing
  hands_health      — platform state change (enable/disable/error)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger("empire.hands")


# ── Types ────────────────────────────────────────────────────────────────

@dataclass
class PublishedListing:
    platform: str
    ok: bool
    url: str | None = None
    listing_id: str | None = None
    basket_impressions: int = 0
    error: str | None = None
    latency_ms: int = 0


@dataclass
class HealthStatus:
    platform: str
    ready: bool
    last_check: float = field(default_factory=time.time)
    error: str | None = None


class Adapter(Protocol):
    """Every platform-specific publisher implements this."""
    platform: str

    async def publish(self, product: dict) -> PublishedListing: ...
    async def health(self) -> HealthStatus: ...


# ── Mock adapter base ────────────────────────────────────────────────────

class MockAdapter:
    """Shared base for the 4 v1 mock adapters. Simulates a ~300-800ms API
    round-trip and returns a mock listing URL. Overrideable by subclasses
    that want different basket_impression profiles.
    """

    def __init__(self, platform: str, base_url_template: str,
                 basket_profile: tuple[int, int] = (5, 80)):
        self.platform = platform
        self.base_url_template = base_url_template
        self._basket_min, self._basket_max = basket_profile

    async def publish(self, product: dict) -> PublishedListing:
        t0 = time.perf_counter()
        # Simulate a realistic API call latency.
        await asyncio.sleep(0.3 + 0.5 * (hash(product.get("name", "")) % 5) / 5)
        listing_id = uuid.uuid4().hex[:10]
        url = self.base_url_template.format(listing_id=listing_id)
        # Mock basket impressions — random-ish but deterministic for tests.
        impressions = self._basket_min + (hash(listing_id) % (self._basket_max - self._basket_min))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return PublishedListing(
            platform=self.platform,
            ok=True,
            url=url,
            listing_id=listing_id,
            basket_impressions=impressions,
            latency_ms=latency_ms,
        )

    async def health(self) -> HealthStatus:
        # Mocks are always healthy — real adapters will actually ping their APIs.
        return HealthStatus(platform=self.platform, ready=True)


class TikTokShopMockAdapter(MockAdapter):
    def __init__(self):
        super().__init__(
            platform="tiktok",
            base_url_template="mock://tiktok-shop/{listing_id}",
            basket_profile=(30, 120),  # TikTok Shop drives strongest impressions
        )


class ShopifyMockAdapter(MockAdapter):
    """Mock adapter. Replace with a Shopify Storefront GraphQL client when
    SHOPIFY_STORE_URL + SHOPIFY_STOREFRONT_TOKEN are set in .env. See
    https://shopify.dev/docs/api/storefront for productCreate mutation."""
    def __init__(self):
        super().__init__(
            platform="shopify",
            base_url_template="mock://anagamastudio.myshopify.com/products/{listing_id}",
            basket_profile=(10, 40),
        )


class EtsyMockAdapter(MockAdapter):
    def __init__(self):
        super().__init__(
            platform="etsy",
            base_url_template="mock://etsy.com/listing/{listing_id}",
            basket_profile=(5, 20),
        )


class InstagramMockAdapter(MockAdapter):
    """Mock adapter for Instagram Live / UGC autoposter. Real integration
    requires Meta Graph API partner approval."""
    def __init__(self):
        super().__init__(
            platform="instagram",
            base_url_template="mock://instagram.com/reel/{listing_id}",
            basket_profile=(8, 35),
        )


# ── Hands orchestrator ───────────────────────────────────────────────────

class Hands:
    """Top-level Hands agent. Holds enabled-state per platform + fans out
    publish() to each enabled adapter in parallel. Emits hands_published /
    hands_health events via the same broadcast callable the Director uses."""

    def __init__(self, broadcast: Callable[[dict], Awaitable[None]] | None = None):
        self.broadcast = broadcast
        self.adapters: dict[str, Adapter] = {
            "tiktok":    TikTokShopMockAdapter(),
            "shopify":   ShopifyMockAdapter(),
            "etsy":      EtsyMockAdapter(),
            "instagram": InstagramMockAdapter(),
        }
        # Default enabled set matches the mockup (TikTok + Shopify ON).
        self.enabled: dict[str, bool] = {
            "tiktok":    True,
            "shopify":   True,
            "etsy":      False,
            "instagram": False,
        }
        # Per-platform last result for /api/hands/state.
        self._last: dict[str, PublishedListing | None] = {p: None for p in self.adapters}

    # ── State surface ────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Snapshot shape for /api/hands/state. Matches what
        DistributionToggles expects on mount."""
        return {
            "platforms": {
                p: {
                    "enabled": self.enabled.get(p, False),
                    "last_publish": self._serialize_last(p),
                    "adapter_type": type(a).__name__,
                }
                for p, a in self.adapters.items()
            }
        }

    def _serialize_last(self, platform: str) -> dict | None:
        last = self._last.get(platform)
        if not last:
            return None
        return {
            "ok": last.ok,
            "url": last.url,
            "listing_id": last.listing_id,
            "basket_impressions": last.basket_impressions,
            "latency_ms": last.latency_ms,
            "error": last.error,
        }

    def set_enabled(self, platform: str, enabled: bool) -> None:
        if platform not in self.adapters:
            raise ValueError(f"unknown platform {platform!r}")
        self.enabled[platform] = enabled
        logger.info("[hands] %s → %s", platform, "enabled" if enabled else "disabled")

    # ── Publish ──────────────────────────────────────────────────────────

    async def publish_all(self, product: dict) -> dict[str, PublishedListing]:
        """Fan out to every enabled platform in parallel. Emits a
        hands_published event per successful publish so the dashboard's
        MetricsStrip can tick BASKETS / GMV in real time."""
        active = [(p, a) for p, a in self.adapters.items() if self.enabled.get(p)]
        if not active:
            logger.info("[hands] publish_all: no enabled platforms, skipping")
            return {}

        async def _one(platform_name: str, adapter: Adapter) -> tuple[str, PublishedListing]:
            try:
                result = await adapter.publish(product)
            except Exception as e:
                logger.exception("[hands] %s publish failed", platform_name)
                result = PublishedListing(
                    platform=platform_name, ok=False, error=str(e)[:200]
                )
            self._last[platform_name] = result
            await self._emit_published(result, product)
            return platform_name, result

        pairs = await asyncio.gather(*(
            _one(p, a) for p, a in active
        ))
        return dict(pairs)

    async def _emit_published(self, result: PublishedListing, product: dict) -> None:
        if not self.broadcast:
            return
        try:
            await self.broadcast({
                "type": "hands_published",
                "platform": result.platform,
                "ok": result.ok,
                "url": result.url,
                "listing_id": result.listing_id,
                "basket_impressions": result.basket_impressions,
                "product_name": product.get("name"),
                "error": result.error,
                "latency_ms": result.latency_ms,
            })
        except Exception:
            logger.exception("[hands] broadcast failed")
