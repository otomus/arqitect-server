<script setup>
import DefaultTheme from 'vitepress/theme'
import { useRoute } from 'vitepress'
import { computed, ref, onMounted, nextTick, watch } from 'vue'

const { Layout } = DefaultTheme
const route = useRoute()
const isHome = computed(() => route.path === '/' || route.path === '/arqitect-server/')
const menuOpen = ref(false)

const navLinks = [
  { text: 'Docs', href: '/arqitect-server/guide/getting-started' },
  { text: 'Architecture', href: '/arqitect-server/architecture/overview' },
  { text: 'Monitoring', href: '/arqitect-server/monitoring' },
  { text: 'Community', href: 'https://otomus.github.io/arqitect-community/' },
  { text: 'Dashboard', href: 'https://otomus.github.io/arqitect-dashboard/' },
]

function initScrollReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible')
          observer.unobserve(entry.target)
        }
      })
    },
    { threshold: 0.15 }
  )

  document.querySelectorAll('.VPFeature, .vp-doc h2, .vp-doc h3, .vp-doc .custom-block, .vp-doc table').forEach((el) => {
    el.classList.add('arq-reveal')
    observer.observe(el)
  })

  // Stagger feature cards
  document.querySelectorAll('.VPFeature').forEach((el, i) => {
    el.style.transitionDelay = `${i * 0.1}s`
  })
}

function initGlitchTitle() {
  const heroName = document.querySelector('.VPHero .name')
  if (!heroName) return
  heroName.classList.add('arq-hero-glitch')
  heroName.setAttribute('data-text', heroName.textContent || '')
}

function initHeroPulse() {
  const hero = document.querySelector('.VPHero')
  if (!hero) return

  // Add a divider line after tagline like community site
  const tagline = hero.querySelector('.tagline')
  if (tagline && !hero.querySelector('.arq-hero-line')) {
    const line = document.createElement('div')
    line.className = 'arq-hero-line'
    tagline.after(line)
  }

  // Add entrance classes
  const name = hero.querySelector('.name')
  const text = hero.querySelector('.text')
  const actions = hero.querySelector('.actions')
  if (name) name.classList.add('arq-hero-entrance')
  if (text) text.classList.add('arq-hero-text-entrance')
  if (tagline) tagline.classList.add('arq-hero-tagline-entrance')
  if (actions) actions.classList.add('arq-hero-actions-entrance')
}

onMounted(() => {
  nextTick(() => {
    initScrollReveal()
    initGlitchTitle()
    initHeroPulse()
  })
})

watch(() => route.path, () => {
  nextTick(() => {
    initScrollReveal()
  })
})
</script>

<template>
  <Layout>
    <template #layout-top>
      <div class="arq-wip-banner">Work in progress — coming soon</div>
      <div class="arq-particles" aria-hidden="true">
        <span v-for="i in 20" :key="i" class="arq-particle" />
      </div>
    </template>

    <template #nav-bar-content-before>
      <nav class="arq-nav">
        <div class="arq-nav-inner">
          <a href="/arqitect-server/" class="arq-nav-logo-link" aria-label="ARQITECT">
            <svg class="arq-nav-logo" viewBox="0 0 220 28" fill="none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="nav-grad" x1="0" y1="0" x2="220" y2="0" gradientUnits="userSpaceOnUse">
                  <stop offset="0%" stop-color="#00d4ff"/>
                  <stop offset="50%" stop-color="#00ff88"/>
                  <stop offset="100%" stop-color="#00d4ff"/>
                </linearGradient>
              </defs>
              <text x="0" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">A<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="0.0s" repeatCount="indefinite"/></text>
              <text x="25" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">R<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="0.4s" repeatCount="indefinite"/></text>
              <text x="50" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">Q<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="0.8s" repeatCount="indefinite"/></text>
              <text x="75" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">I<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="1.2s" repeatCount="indefinite"/></text>
              <text x="100" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">T<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="1.6s" repeatCount="indefinite"/></text>
              <text x="125" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">E<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="2.0s" repeatCount="indefinite"/></text>
              <text x="150" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">C<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="2.4s" repeatCount="indefinite"/></text>
              <text x="175" y="20" font-family="Orbitron,sans-serif" font-size="18" font-weight="900" fill="none" stroke="url(#nav-grad)" stroke-width="1" stroke-dasharray="80" stroke-dashoffset="80">T<animate attributeName="stroke-dashoffset" values="80;0;0;80" keyTimes="0;0.085;0.681;1" dur="4.7s" begin="2.8s" repeatCount="indefinite"/></text>
            </svg>
          </a>
          <button class="arq-nav-toggle" aria-label="Menu" @click="menuOpen = !menuOpen">&#9776;</button>
          <div class="arq-nav-links" :class="{ open: menuOpen }">
            <a v-for="link in navLinks" :key="link.text" :href="link.href">{{ link.text }}</a>
          </div>
        </div>
      </nav>
    </template>

    <template #layout-bottom>
      <footer class="arq-footer">
        <div class="arq-footer-links">
          <a href="https://github.com/otomus/arqitect-server" target="_blank" aria-label="GitHub">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
          </a>
          <a href="https://discord.gg/AYjJgWuS" target="_blank" aria-label="Discord">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 00-4.885-1.515.074.074 0 00-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 00-5.487 0 12.64 12.64 0 00-.617-1.25.077.077 0 00-.079-.037A19.736 19.736 0 003.677 4.37a.07.07 0 00-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 00.031.057 19.9 19.9 0 005.993 3.03.078.078 0 00.084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 00-.041-.106 13.107 13.107 0 01-1.872-.892.077.077 0 01-.008-.128 10.2 10.2 0 00.372-.292.074.074 0 01.077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 01.078.01c.12.098.246.198.373.292a.077.077 0 01-.006.127 12.299 12.299 0 01-1.873.892.077.077 0 00-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 00.084.028 19.839 19.839 0 006.002-3.03.077.077 0 00.032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 00-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z"/></svg>
          </a>
          <a href="https://join.slack.com/t/arqitectworkspace/shared_invite/zt-3t6zqf442-yMtBciurcPd_J66gZ1l1wA" target="_blank" aria-label="Slack">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M5.042 15.165a2.528 2.528 0 01-2.52 2.523A2.528 2.528 0 010 15.165a2.527 2.527 0 012.522-2.52h2.52v2.52zm1.271 0a2.527 2.527 0 012.521-2.52 2.527 2.527 0 012.521 2.52v6.313A2.528 2.528 0 018.834 24a2.528 2.528 0 01-2.521-2.522v-6.313zM8.834 5.042a2.528 2.528 0 01-2.521-2.52A2.528 2.528 0 018.834 0a2.528 2.528 0 012.521 2.522v2.52H8.834zm0 1.271a2.528 2.528 0 012.521 2.521 2.528 2.528 0 01-2.521 2.521H2.522A2.528 2.528 0 010 8.834a2.528 2.528 0 012.522-2.521h6.312zM18.956 8.834a2.528 2.528 0 012.522-2.521A2.528 2.528 0 0124 8.834a2.528 2.528 0 01-2.522 2.521h-2.522V8.834zm-1.27 0a2.528 2.528 0 01-2.523 2.521 2.527 2.527 0 01-2.52-2.521V2.522A2.527 2.527 0 0115.163 0a2.528 2.528 0 012.523 2.522v6.312zM15.163 18.956a2.528 2.528 0 012.523 2.522A2.528 2.528 0 0115.163 24a2.527 2.527 0 01-2.52-2.522v-2.522h2.52zm0-1.27a2.527 2.527 0 01-2.52-2.523 2.526 2.526 0 012.52-2.52h6.315A2.528 2.528 0 0124 15.163a2.528 2.528 0 01-2.522 2.523h-6.315z"/></svg>
          </a>
        </div>
        <span>Arqitect &mdash; guardians, not rulers</span>
      </footer>
    </template>
  </Layout>
</template>
