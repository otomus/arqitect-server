<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const step = ref(-1)
const isVisible = ref(false)
let interval: ReturnType<typeof setInterval> | undefined
let timeout: ReturnType<typeof setTimeout> | undefined

const TOTAL_STEPS = 9
const STEP_DURATION = 1800

const nerves = ['image-gen', 'web-search', 'calendar', 'email', 'translate']
const selectedNerve = 2

const steps = [
  { label: 'User sends a message' },
  { label: 'Brain receives the message' },
  { label: 'Brain searches for the right nerve' },
  { label: 'Nerve selected — others stand down' },
  { label: 'Nerve invokes a tool' },
  { label: 'Tool returns result to Brain' },
  { label: 'Brain routes to Communication nerve' },
  { label: 'Communication nerve processes response' },
  { label: 'Response returned to user' },
]

function startAnimation() {
  step.value = 0
  interval = setInterval(() => {
    if (step.value < TOTAL_STEPS - 1) {
      step.value++
    } else {
      clearInterval(interval)
      timeout = setTimeout(() => startAnimation(), 3000)
    }
  }, STEP_DURATION)
}

onMounted(() => {
  const observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting && !isVisible.value) {
        isVisible.value = true
        startAnimation()
        observer.disconnect()
      }
    },
    { threshold: 0.3 }
  )
  if (containerRef.value) observer.observe(containerRef.value)
  onUnmounted(() => {
    observer.disconnect()
    clearInterval(interval)
    clearTimeout(timeout)
  })
})
</script>

<template>
  <div class="flow-wrapper" ref="containerRef">
    <!-- Step indicator -->
    <div class="step-label" :class="{ visible: isVisible }">
      <span class="step-text" :key="step">{{ steps[step]?.label ?? '' }}</span>
    </div>

    <svg
      class="flow-diagram"
      :class="{ visible: isVisible }"
      viewBox="0 0 1000 520"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="gl-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00d4ff" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="gl-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00ff88" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="gl-orange" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff6a00" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="gl-magenta" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff00aa" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="gl-yellow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#facc15" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <marker id="arr-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff" />
        </marker>
        <marker id="arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88" />
        </marker>
        <marker id="arr-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6a00" />
        </marker>
        <marker id="arr-yellow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#facc15" />
        </marker>
        <marker id="arr-magenta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff00aa" />
        </marker>
      </defs>

      <!-- ===== USER NODE (left) ===== -->
      <g :class="['node', { active: step >= 0 }]">
        <circle cx="80" cy="230" r="48" fill="rgba(0,212,255,0.05)" stroke="#00d4ff" stroke-width="2" :filter="step === 0 || step === 8 ? 'url(#gl-cyan)' : ''" />
        <text x="80" y="222" text-anchor="middle" fill="#00d4ff" font-size="28">👤</text>
        <text x="80" y="248" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="13" font-weight="700" letter-spacing="0.12em">USER</text>
      </g>

      <!-- ===== ARROW: User → Brain (step 0) ===== -->
      <line
        x1="132" y1="230" x2="240" y2="230"
        :class="['arrow', { active: step >= 0 }]"
        stroke="#00d4ff" stroke-width="2" marker-end="url(#arr-cyan)"
      />
      <circle v-if="step === 0" r="5" fill="#00d4ff" filter="url(#gl-cyan)" class="particle">
        <animateMotion dur="0.8s" repeatCount="indefinite" path="M132,230 L240,230" />
      </circle>

      <!-- ===== BRAIN NODE (center-left) ===== -->
      <g :class="['node', { active: step >= 1 }]">
        <circle cx="300" cy="230" r="56" fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="2.5" :filter="step >= 1 && step <= 2 || step >= 5 ? 'url(#gl-cyan)' : ''" />
        <circle cx="300" cy="230" r="42" fill="none" stroke="#00d4ff" stroke-width="0.8" stroke-opacity="0.3" stroke-dasharray="4 5" :class="{ spinning: step >= 1 }" />
        <text x="300" y="236" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="18" font-weight="900" letter-spacing="0.22em">BRAIN</text>
      </g>

      <!-- ===== ARROWS: Brain → Nerves (step 2) ===== -->
      <template v-for="(nerve, i) in nerves" :key="'line-' + nerve">
        <line
          :x1="360" :y1="230"
          :x2="520" :y2="75 + i * 80"
          :class="['arrow', 'nerve-arrow', { active: step >= 2, stale: step >= 3 && i !== selectedNerve, selected: step >= 3 && i === selectedNerve }]"
          :stroke="i === selectedNerve ? '#00ff88' : '#00d4ff'"
          stroke-width="2"
          :marker-end="i === selectedNerve ? 'url(#arr-green)' : 'url(#arr-cyan)'"
          :style="{ transitionDelay: (i * 0.08) + 's' }"
        />
        <circle v-if="step === 2" r="4" fill="#00d4ff" filter="url(#gl-cyan)" class="particle">
          <animateMotion :dur="(0.6 + i * 0.1) + 's'" repeatCount="indefinite" :path="'M360,230 L520,' + (75 + i * 80)" />
        </circle>
      </template>

      <!-- ===== NERVE NODES ===== -->
      <g v-for="(nerve, i) in nerves" :key="'nerve-' + nerve"
        :class="['node', 'nerve-node', { active: step >= 2, stale: step >= 3 && i !== selectedNerve, selected: step >= 3 && i === selectedNerve }]"
        :style="{ transitionDelay: (i * 0.08) + 's' }"
      >
        <rect
          :x="520" :y="52 + i * 80" width="160" height="48" rx="5"
          :fill="step >= 3 && i === selectedNerve ? 'rgba(0,255,136,0.08)' : 'rgba(0,212,255,0.04)'"
          :stroke="step >= 3 && i === selectedNerve ? '#00ff88' : '#00d4ff'"
          stroke-width="1.5"
          :filter="step >= 3 && i === selectedNerve ? 'url(#gl-green)' : ''"
        />
        <text
          :x="600" :y="82 + i * 80"
          text-anchor="middle"
          :fill="step >= 3 && i !== selectedNerve ? '#4a4a6a' : (i === selectedNerve && step >= 3 ? '#00ff88' : '#00d4ff')"
          font-family="'Share Tech Mono',monospace" font-size="15"
        >{{ nerve }}</text>
      </g>

      <!-- ===== ARROW: Selected Nerve → Tool (step 4) ===== -->
      <line
        x1="680" :y1="76 + selectedNerve * 80"
        x2="790" :y2="76 + selectedNerve * 80"
        :class="['arrow', { active: step >= 4 }]"
        stroke="#facc15" stroke-width="2" marker-end="url(#arr-yellow)"
      />
      <circle v-if="step === 4" r="5" fill="#facc15" filter="url(#gl-yellow)" class="particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" :path="'M680,' + (76 + selectedNerve * 80) + ' L790,' + (76 + selectedNerve * 80)" />
      </circle>

      <!-- ===== TOOL NODE ===== -->
      <g :class="['node', { active: step >= 4 }]">
        <rect
          x="790" :y="52 + selectedNerve * 80" width="140" height="48" rx="5"
          fill="rgba(250,204,21,0.06)" stroke="#facc15" stroke-width="1.5"
          :filter="step === 4 ? 'url(#gl-yellow)' : ''"
        />
        <text x="860" :y="72 + selectedNerve * 80" text-anchor="middle" fill="#facc15" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.1em">TOOL</text>
        <text x="860" :y="90 + selectedNerve * 80" text-anchor="middle" fill="rgba(250,204,21,0.6)" font-family="'Share Tech Mono',monospace" font-size="12">mcp:search</text>
      </g>

      <!-- ===== ARROW: Tool → Brain (step 5) ===== -->
      <line
        x1="790" :y1="76 + selectedNerve * 80"
        x2="680" :y2="76 + selectedNerve * 80"
        :class="['arrow', 'return-arrow', { active: step >= 5 }]"
        stroke="#ff6a00" stroke-width="2" marker-end="url(#arr-orange)"
        transform="translate(0, 18)"
      />
      <line
        x1="520" :y1="94 + selectedNerve * 80"
        x2="360" y2="248"
        :class="['arrow', 'return-arrow', { active: step >= 5 }]"
        stroke="#ff6a00" stroke-width="2" marker-end="url(#arr-orange)"
      />
      <circle v-if="step === 5" r="5" fill="#ff6a00" filter="url(#gl-orange)" class="particle">
        <animateMotion dur="1.2s" repeatCount="indefinite" :path="'M790,' + (94 + selectedNerve * 80) + ' L600,' + (94 + selectedNerve * 80) + ' L360,248'" />
      </circle>

      <!-- ===== ARROW: Brain → Communication Nerve (step 6) ===== -->
      <line
        x1="300" y1="290" x2="300" y2="385"
        :class="['arrow', { active: step >= 6 }]"
        stroke="#ff00aa" stroke-width="2" marker-end="url(#arr-magenta)"
      />
      <circle v-if="step === 6" r="5" fill="#ff00aa" filter="url(#gl-magenta)" class="particle">
        <animateMotion dur="0.6s" repeatCount="indefinite" path="M300,290 L300,385" />
      </circle>

      <!-- ===== COMMUNICATION NERVE NODE ===== -->
      <g :class="['node', { active: step >= 6 }]">
        <rect
          x="210" y="390" width="180" height="54" rx="5"
          fill="rgba(255,0,170,0.06)" stroke="#ff00aa" stroke-width="1.5"
          :filter="step >= 7 ? 'url(#gl-magenta)' : ''"
        />
        <text x="300" y="414" text-anchor="middle" fill="#ff00aa" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.08em">COMMUNICATION</text>
        <text x="300" y="434" text-anchor="middle" fill="rgba(255,0,170,0.6)" font-family="'Share Tech Mono',monospace" font-size="13">
          {{ step === 7 ? 'processing...' : 'nerve' }}
        </text>
      </g>

      <!-- ===== ARROW: Comm Nerve → Brain (step 7) ===== -->
      <line
        x1="300" y1="385" x2="300" y2="290"
        :class="['arrow', 'return-arrow', { active: step >= 7 }]"
        stroke="#ff00aa" stroke-width="2" marker-end="url(#arr-magenta)"
        transform="translate(36, 0)"
      />
      <circle v-if="step === 7" r="5" fill="#ff00aa" filter="url(#gl-magenta)" class="particle">
        <animateMotion dur="0.6s" repeatCount="indefinite" path="M336,385 L336,290" />
      </circle>

      <!-- ===== ARROW: Brain → User (step 8) ===== -->
      <line
        x1="240" y1="230" x2="132" y2="230"
        :class="['arrow', 'return-arrow', { active: step >= 8 }]"
        stroke="#00ff88" stroke-width="2" marker-end="url(#arr-green)"
        transform="translate(0, 20)"
      />
      <circle v-if="step === 8" r="5" fill="#00ff88" filter="url(#gl-green)" class="particle">
        <animateMotion dur="0.8s" repeatCount="indefinite" path="M240,250 L132,250" />
      </circle>

      <!-- ===== Step progress dots ===== -->
      <g transform="translate(320, 505)">
        <circle
          v-for="(s, i) in steps" :key="'dot-' + i"
          :cx="i * 22" cy="0" r="4"
          :fill="i === step ? '#00d4ff' : (i < step ? 'rgba(0,212,255,0.4)' : 'rgba(0,212,255,0.1)')"
          :class="{ 'dot-active': i === step }"
        />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.flow-wrapper {
  max-width: 1000px;
  margin: 2rem auto 0;
  padding: 0 1rem;
}

.step-label {
  text-align: center;
  min-height: 36px;
  margin-bottom: 0.5rem;
  opacity: 0;
  transition: opacity 0.6s;
}

.step-label.visible {
  opacity: 1;
}

.step-text {
  display: inline-block;
  font-family: 'Share Tech Mono', monospace;
  font-size: 1.1rem;
  color: rgba(0, 212, 255, 0.8);
  letter-spacing: 0.05em;
  animation: text-appear 0.4s ease forwards;
}

@keyframes text-appear {
  0% { opacity: 0; transform: translateY(6px); filter: blur(4px); }
  100% { opacity: 1; transform: translateY(0); filter: blur(0); }
}

.flow-diagram {
  width: 100%;
  height: auto;
  opacity: 0;
  transition: opacity 0.8s ease;
}

.flow-diagram.visible {
  opacity: 1;
}

.node {
  opacity: 0.15;
  transition: opacity 0.5s ease;
}

.node.active {
  opacity: 1;
}

.nerve-node.stale {
  opacity: 0.2;
}

.nerve-node.selected {
  opacity: 1;
}

.arrow {
  opacity: 0;
  stroke-dasharray: 400;
  stroke-dashoffset: 400;
  transition: opacity 0.4s ease, stroke-dashoffset 0.6s ease;
}

.arrow.active {
  opacity: 0.6;
  stroke-dashoffset: 0;
}

.nerve-arrow.stale {
  opacity: 0.08 !important;
}

.nerve-arrow.selected {
  opacity: 0.8 !important;
}

.return-arrow.active {
  stroke-dasharray: 8 5;
  stroke-dashoffset: 0;
  animation: dash-flow 0.8s linear infinite;
}

@keyframes dash-flow {
  to { stroke-dashoffset: -26; }
}

.particle {
  opacity: 0.9;
}

.spinning {
  transform-origin: 300px 230px;
  animation: spin 8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.dot-active {
  animation: dot-pulse 1s ease-in-out infinite;
}

@keyframes dot-pulse {
  0%, 100% { r: 4; }
  50% { r: 6; }
}

@media (max-width: 640px) {
  .flow-wrapper {
    margin: 1rem auto 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .flow-diagram {
    min-width: 600px;
  }
  .step-text {
    font-size: 0.85rem;
  }
}
</style>
