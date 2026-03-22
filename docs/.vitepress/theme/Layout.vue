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
        Arqitect &mdash; guardians, not rulers
      </footer>
    </template>
  </Layout>
</template>
