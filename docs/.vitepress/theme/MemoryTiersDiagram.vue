<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const step = ref(-1)
const isVisible = ref(false)
let interval: ReturnType<typeof setInterval> | undefined
let timeout: ReturnType<typeof setTimeout> | undefined

const TOTAL_STEPS = 8
const STEP_DURATION = 2000

const steps = [
  { label: 'A task arrives' },
  { label: 'Hot memory holds the live session (Redis)' },
  { label: 'Brain checks warm memory — "have I seen this before?"' },
  { label: 'Warm memory stores episodes and task history (SQLite)' },
  { label: 'Task executes — result recorded as an episode' },
  { label: 'Knowledge extracted from episodes to cold memory' },
  { label: 'Cold memory persists nerves, tools, users, personality (SQLite)' },
  { label: 'System restarts — cold memory survives, hot is rebuilt' },
]

function startAnimation(): void {
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
      viewBox="0 0 1000 480"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <!-- Glow filters -->
        <filter id="mt-gl-orange" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff6a00" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="mt-gl-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00ff88" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="mt-gl-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00d4ff" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="mt-gl-red" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff4444" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="mt-gl-yellow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#facc15" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <!-- Arrow markers -->
        <marker id="mt-arr-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6a00" />
        </marker>
        <marker id="mt-arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88" />
        </marker>
        <marker id="mt-arr-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff" />
        </marker>
        <marker id="mt-arr-yellow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#facc15" />
        </marker>
        <marker id="mt-arr-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff4444" />
        </marker>
      </defs>

      <!-- ===== TASK NODE (far left) ===== -->
      <g :class="['node', { active: step >= 0 }]">
        <circle cx="60" cy="200" r="32" fill="rgba(250,204,21,0.05)" stroke="#facc15" stroke-width="2" :filter="step === 0 ? 'url(#mt-gl-yellow)' : ''" />
        <text x="60" y="196" text-anchor="middle" fill="#facc15" font-size="22">📋</text>
        <text x="60" y="214" text-anchor="middle" fill="#facc15" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.1em">TASK</text>
      </g>

      <!-- ===== ARROW: Task → Hot Memory (step 0→1) ===== -->
      <line
        x1="94" y1="200" x2="158" y2="200"
        :class="['arrow', { active: step >= 1 }]"
        stroke="#ff6a00" stroke-width="2" marker-end="url(#mt-arr-orange)"
      />
      <circle v-if="step === 0 || step === 1" r="4" fill="#facc15" filter="url(#mt-gl-yellow)" class="particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M94,200 L158,200" />
      </circle>

      <!-- ===== HOT MEMORY (Redis) ===== -->
      <g :class="['node', { active: step >= 1 }]">
        <rect
          x="160" y="130" width="180" height="140" rx="8"
          fill="rgba(255,106,0,0.06)" stroke="#ff6a00" stroke-width="2"
          :filter="step === 1 ? 'url(#mt-gl-orange)' : ''"
        />
        <text x="250" y="158" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="14" font-weight="900" letter-spacing="0.15em">HOT MEMORY</text>
        <text x="250" y="178" text-anchor="middle" fill="rgba(255,106,0,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">Redis</text>
        <line x1="180" y1="188" x2="320" y2="188" stroke="rgba(255,106,0,0.2)" stroke-width="1" />
        <text x="250" y="208" text-anchor="middle" fill="rgba(255,106,0,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">session</text>
        <text x="250" y="226" text-anchor="middle" fill="rgba(255,106,0,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">conversation</text>
        <text x="250" y="244" text-anchor="middle" fill="rgba(255,106,0,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">state</text>
        <!-- Ephemeral indicator -->
        <text x="250" y="262" text-anchor="middle" fill="rgba(255,106,0,0.35)" font-family="'Share Tech Mono',monospace" font-size="10" font-style="italic">ephemeral</text>
      </g>

      <!-- ===== ARROW: Hot → Warm query (step 2) ===== -->
      <line
        x1="342" y1="200" x2="418" y2="200"
        :class="['arrow', { active: step >= 2 }]"
        stroke="#00ff88" stroke-width="2" marker-end="url(#mt-arr-green)"
      />
      <circle v-if="step === 2" r="4" fill="#00ff88" filter="url(#mt-gl-green)" class="particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M342,200 L418,200" />
      </circle>
      <!-- Query label -->
      <text
        x="380" y="188"
        :class="['node', { active: step >= 2 }]"
        text-anchor="middle" fill="rgba(0,255,136,0.6)" font-family="'Share Tech Mono',monospace" font-size="10"
      >seen this?</text>

      <!-- ===== WARM MEMORY (SQLite Episodes) ===== -->
      <g :class="['node', { active: step >= 3 }]">
        <rect
          x="420" y="130" width="180" height="140" rx="8"
          fill="rgba(0,255,136,0.05)" stroke="#00ff88" stroke-width="2"
          :filter="step === 3 || step === 4 ? 'url(#mt-gl-green)' : ''"
        />
        <text x="510" y="158" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="14" font-weight="900" letter-spacing="0.12em">WARM MEMORY</text>
        <text x="510" y="178" text-anchor="middle" fill="rgba(0,255,136,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">SQLite Episodes</text>
        <line x1="440" y1="188" x2="580" y2="188" stroke="rgba(0,255,136,0.2)" stroke-width="1" />
        <text x="510" y="208" text-anchor="middle" fill="rgba(0,255,136,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">episodes</text>
        <text x="510" y="226" text-anchor="middle" fill="rgba(0,255,136,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">task history</text>
        <text x="510" y="244" text-anchor="middle" fill="rgba(0,255,136,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">results</text>
        <text x="510" y="262" text-anchor="middle" fill="rgba(0,255,136,0.35)" font-family="'Share Tech Mono',monospace" font-size="10" font-style="italic">survives restarts</text>
      </g>

      <!-- ===== ARROW: Execution result → Warm (step 4, curves down and back) ===== -->
      <path
        d="M510,270 L510,320 Q510,340 490,340 L490,340"
        :class="['arrow', 'return-arrow', { active: step >= 4 }]"
        fill="none" stroke="#00ff88" stroke-width="2"
      />
      <text
        x="510" y="360"
        :class="['node', { active: step >= 4 }]"
        text-anchor="middle" fill="rgba(0,255,136,0.6)" font-family="'Share Tech Mono',monospace" font-size="10"
      >record episode</text>
      <circle v-if="step === 4" r="4" fill="#00ff88" filter="url(#mt-gl-green)" class="particle">
        <animateMotion dur="0.8s" repeatCount="indefinite" path="M510,320 L510,280" />
      </circle>

      <!-- ===== ARROW: Warm → Cold (step 5) ===== -->
      <line
        x1="602" y1="200" x2="678" y2="200"
        :class="['arrow', { active: step >= 5 }]"
        stroke="#00d4ff" stroke-width="2" marker-end="url(#mt-arr-cyan)"
      />
      <circle v-if="step === 5" r="4" fill="#00d4ff" filter="url(#mt-gl-cyan)" class="particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M602,200 L678,200" />
      </circle>
      <text
        x="640" y="188"
        :class="['node', { active: step >= 5 }]"
        text-anchor="middle" fill="rgba(0,212,255,0.6)" font-family="'Share Tech Mono',monospace" font-size="10"
      >extract</text>

      <!-- ===== COLD MEMORY (SQLite Knowledge) — largest node ===== -->
      <g :class="['node', { active: step >= 6 }]">
        <rect
          x="680" y="105" width="260" height="190" rx="10"
          fill="rgba(0,212,255,0.06)" stroke="#00d4ff" stroke-width="2.5"
          :filter="step >= 6 ? 'url(#mt-gl-cyan)' : ''"
        />
        <!-- Decorative inner border -->
        <rect
          x="692" y="117" width="236" height="166" rx="6"
          fill="none" stroke="#00d4ff" stroke-width="0.6" stroke-opacity="0.2" stroke-dasharray="4 5"
        />
        <text x="810" y="142" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="15" font-weight="900" letter-spacing="0.15em">COLD MEMORY</text>
        <text x="810" y="162" text-anchor="middle" fill="rgba(0,212,255,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">SQLite Knowledge</text>
        <line x1="705" y1="172" x2="915" y2="172" stroke="rgba(0,212,255,0.2)" stroke-width="1" />
        <text x="760" y="192" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">nerves</text>
        <text x="860" y="192" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">tools</text>
        <text x="760" y="214" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">users</text>
        <text x="860" y="214" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">personality</text>
        <text x="810" y="236" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">permissions</text>
        <text x="810" y="258" text-anchor="middle" fill="rgba(0,212,255,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">community catalog</text>
        <text x="810" y="280" text-anchor="middle" fill="rgba(0,212,255,0.35)" font-family="'Share Tech Mono',monospace" font-size="10" font-style="italic">permanent — survives everything</text>
      </g>

      <!-- ===== RESTART INDICATOR (step 7) ===== -->
      <g :class="['node', { active: step >= 7 }]">
        <!-- Restart flash over hot memory -->
        <rect
          x="160" y="130" width="180" height="140" rx="8"
          fill="rgba(255,68,68,0.12)" stroke="#ff4444" stroke-width="2" stroke-dasharray="6 4"
          :filter="step === 7 ? 'url(#mt-gl-red)' : ''"
        />
        <text x="250" y="305" text-anchor="middle" fill="#ff4444" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.1em">WIPED ON RESTART</text>

        <!-- Cold memory persistence badge -->
        <rect x="746" y="310" width="128" height="30" rx="4" fill="rgba(0,212,255,0.1)" stroke="#00d4ff" stroke-width="1.5" />
        <text x="810" y="330" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.08em">PERSISTS</text>

        <!-- Warm memory persistence badge -->
        <rect x="476" y="310" width="68" height="24" rx="4" fill="rgba(0,255,136,0.08)" stroke="#00ff88" stroke-width="1" />
        <text x="510" y="327" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="9" font-weight="700">PERSISTS</text>
      </g>

      <!-- ===== TIER LABELS (bottom) ===== -->
      <g :class="['node', { active: step >= 1 }]">
        <text x="250" y="400" text-anchor="middle" fill="rgba(255,106,0,0.4)" font-family="'Orbitron',sans-serif" font-size="10" letter-spacing="0.2em">TIER 1</text>
      </g>
      <g :class="['node', { active: step >= 3 }]">
        <text x="510" y="400" text-anchor="middle" fill="rgba(0,255,136,0.4)" font-family="'Orbitron',sans-serif" font-size="10" letter-spacing="0.2em">TIER 2</text>
      </g>
      <g :class="['node', { active: step >= 6 }]">
        <text x="810" y="400" text-anchor="middle" fill="rgba(0,212,255,0.4)" font-family="'Orbitron',sans-serif" font-size="10" letter-spacing="0.2em">TIER 3</text>
      </g>

      <!-- ===== Step progress dots ===== -->
      <g transform="translate(412, 450)">
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
  animation: mt-text-appear 0.4s ease forwards;
}

@keyframes mt-text-appear {
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

.return-arrow.active {
  stroke-dasharray: 8 5;
  stroke-dashoffset: 0;
  animation: mt-dash-flow 0.8s linear infinite;
}

@keyframes mt-dash-flow {
  to { stroke-dashoffset: -26; }
}

.particle {
  opacity: 0.9;
}

.dot-active {
  animation: mt-dot-pulse 1s ease-in-out infinite;
}

@keyframes mt-dot-pulse {
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
