<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const containerRef = ref<HTMLElement | null>(null)
const step = ref(-1)
const isVisible = ref(false)
let interval: ReturnType<typeof setInterval> | undefined
let timeout: ReturnType<typeof setTimeout> | undefined

const TOTAL_STEPS = 8
const STEP_DURATION = 2200

const steps = [
  { label: 'Brain enters dream state — reflecting on nerves' },
  { label: 'Brain selects "calendar" nerve for review' },
  { label: 'Checking the community for calendar nerve updates' },
  { label: 'Qualification score: 72% — below 95% threshold' },
  { label: 'Dream cycle begins — updating prompts' },
  { label: 'Tuning adapters for the target model' },
  { label: 'Evaluating nerve quality' },
  { label: 'Contributing improved nerve back to the community' },
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
    { threshold: 0.2 }
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
  <div class="dream-wrapper" ref="containerRef">
    <div class="step-label" :class="{ visible: isVisible }">
      <span class="step-text" :key="step">{{ steps[step]?.label ?? '' }}</span>
    </div>

    <svg
      class="dream-diagram"
      :class="{ visible: isVisible }"
      viewBox="0 0 1000 540"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <filter id="d-gl-cyan" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00d4ff" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="d-gl-purple" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feFlood flood-color="#a855f7" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="d-gl-green" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#00ff88" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="d-gl-orange" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#ff6a00" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <filter id="d-gl-yellow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="5" result="b" />
          <feFlood flood-color="#facc15" flood-opacity="0.6" />
          <feComposite in2="b" operator="in" />
          <feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>

        <marker id="d-arr-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00d4ff" />
        </marker>
        <marker id="d-arr-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#a855f7" />
        </marker>
        <marker id="d-arr-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#00ff88" />
        </marker>
        <marker id="d-arr-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6a00" />
        </marker>
      </defs>

      <!-- ===== BRAIN NODE (sleeping) ===== -->
      <g :class="['d-node', { active: step >= 0 }]">
        <circle cx="110" cy="220" r="68" fill="none" stroke="#a855f7" stroke-width="1" stroke-opacity="0.15" :class="{ 'dream-aura': step >= 0 }" />
        <circle cx="110" cy="220" r="56" fill="rgba(168,85,247,0.04)" stroke="#a855f7" stroke-width="2" :filter="step >= 0 ? 'url(#d-gl-purple)' : ''" />
        <circle cx="110" cy="220" r="42" fill="none" stroke="#a855f7" stroke-width="0.6" stroke-opacity="0.3" stroke-dasharray="4 5" :class="{ spinning: step >= 0 }" />
        <text x="110" y="214" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="17" font-weight="900" letter-spacing="0.18em">BRAIN</text>
        <text x="110" y="238" text-anchor="middle" fill="rgba(168,85,247,0.5)" font-family="'Share Tech Mono',monospace" font-size="13">
          {{ step >= 0 ? '💤 dreaming' : '' }}
        </text>
      </g>

      <!-- ===== Stale nerves floating around brain ===== -->
      <g :class="['d-node', 'nerve-cloud', { active: step >= 0 }]">
        <rect x="40" y="310" width="65" height="28" rx="4" fill="rgba(0,212,255,0.03)" stroke="rgba(0,212,255,0.15)" stroke-width="1" :class="{ stale: step >= 1 }" />
        <text x="73" y="328" text-anchor="middle" fill="rgba(0,212,255,0.3)" font-family="'Share Tech Mono',monospace" font-size="11">email</text>

        <rect x="115" y="316" width="80" height="28" rx="4" fill="rgba(0,212,255,0.03)" stroke="rgba(0,212,255,0.15)" stroke-width="1" :class="{ stale: step >= 1 }" />
        <text x="155" y="334" text-anchor="middle" fill="rgba(0,212,255,0.3)" font-family="'Share Tech Mono',monospace" font-size="11">translate</text>

        <rect x="30" y="110" width="85" height="28" rx="4" fill="rgba(0,212,255,0.03)" stroke="rgba(0,212,255,0.15)" stroke-width="1" :class="{ stale: step >= 1 }" />
        <text x="73" y="128" text-anchor="middle" fill="rgba(0,212,255,0.3)" font-family="'Share Tech Mono',monospace" font-size="11">web-search</text>

        <rect x="125" y="118" width="75" height="28" rx="4" fill="rgba(0,212,255,0.03)" stroke="rgba(0,212,255,0.15)" stroke-width="1" :class="{ stale: step >= 1 }" />
        <text x="163" y="136" text-anchor="middle" fill="rgba(0,212,255,0.3)" font-family="'Share Tech Mono',monospace" font-size="11">image-gen</text>
      </g>

      <!-- ===== Selected nerve: calendar (step 1) ===== -->
      <g :class="['d-node', { active: step >= 1 }]">
        <line x1="170" y1="220" x2="260" y2="220" :class="['d-arrow', { active: step >= 1 }]" stroke="#a855f7" stroke-width="2" marker-end="url(#d-arr-purple)" />
        <circle v-if="step === 1" r="5" fill="#a855f7" filter="url(#d-gl-purple)" class="d-particle">
          <animateMotion dur="0.7s" repeatCount="indefinite" path="M170,220 L260,220" />
        </circle>
        <rect
          x="265" y="196" width="150" height="50" rx="5"
          :fill="step >= 1 ? 'rgba(0,255,136,0.06)' : 'rgba(0,212,255,0.04)'"
          :stroke="step >= 1 ? '#00ff88' : '#00d4ff'"
          stroke-width="1.5"
          :filter="step >= 1 && step < 4 ? 'url(#d-gl-green)' : ''"
        />
        <text x="340" y="218" text-anchor="middle" fill="#00ff88" font-family="'Orbitron',sans-serif" font-size="12" font-weight="700" letter-spacing="0.08em">CALENDAR</text>
        <text x="340" y="236" text-anchor="middle" fill="rgba(0,255,136,0.5)" font-family="'Share Tech Mono',monospace" font-size="13">nerve</text>
      </g>

      <!-- ===== Arrow: Calendar → Community (step 2) ===== -->
      <line
        x1="415" y1="220" x2="510" y2="220"
        :class="['d-arrow', { active: step >= 2 }]"
        stroke="#00d4ff" stroke-width="2" marker-end="url(#d-arr-cyan)"
      />
      <circle v-if="step === 2" r="5" fill="#00d4ff" filter="url(#d-gl-cyan)" class="d-particle">
        <animateMotion dur="0.7s" repeatCount="indefinite" path="M415,220 L510,220" />
      </circle>

      <!-- ===== Community node ===== -->
      <g :class="['d-node', { active: step >= 2 }]">
        <circle cx="570" cy="220" r="48" fill="rgba(0,212,255,0.04)" stroke="#00d4ff" stroke-width="1.5" :filter="step === 2 ? 'url(#d-gl-cyan)' : ''" />
        <text x="570" y="212" text-anchor="middle" fill="#00d4ff" font-size="24">🌐</text>
        <text x="570" y="234" text-anchor="middle" fill="#00d4ff" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.08em">COMMUNITY</text>
      </g>

      <!-- Arrow: Community → back -->
      <line
        x1="520" y1="238" x2="418" y2="238"
        :class="['d-arrow', 'd-return', { active: step >= 2 }]"
        stroke="#00d4ff" stroke-width="1.5" stroke-opacity="0.4"
      />

      <!-- ===== Qualification check (step 3) ===== -->
      <g :class="['d-node', { active: step >= 3 }]">
        <line x1="340" y1="250" x2="340" y2="320" :class="['d-arrow', { active: step >= 3 }]" stroke="#ff6a00" stroke-width="2" marker-end="url(#d-arr-orange)" />

        <rect x="255" y="325" width="170" height="65" rx="5" fill="rgba(255,106,0,0.05)" stroke="#ff6a00" stroke-width="1.5" :filter="step === 3 ? 'url(#d-gl-orange)' : ''" />
        <text x="340" y="348" text-anchor="middle" fill="#ff6a00" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.08em">QUALIFICATION</text>
        <text x="340" y="368" text-anchor="middle" fill="rgba(255,106,0,0.7)" font-family="'Share Tech Mono',monospace" font-size="14">
          {{ step >= 3 ? 'score: 72%' : '' }}
        </text>
        <text x="340" y="385" text-anchor="middle" fill="#ff6a00" font-family="'Share Tech Mono',monospace" font-size="11" opacity="0.6">
          {{ step >= 3 ? '< 95% → needs tuning' : '' }}
        </text>
      </g>

      <!-- ===== Dream cycle box ===== -->
      <rect
        x="470" y="290" width="500" height="180" rx="8"
        fill="none"
        :stroke="step >= 4 ? '#a855f7' : 'rgba(168,85,247,0.1)'"
        stroke-width="2"
        stroke-dasharray="8 4"
        :class="['dream-box', { active: step >= 4 }]"
      />
      <text x="720" y="315" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="11" font-weight="700" letter-spacing="0.18em" :opacity="step >= 4 ? 1 : 0.2">
        DREAM CYCLE
      </text>

      <!-- Arrow: qualification → dream cycle -->
      <line
        x1="425" y1="358" x2="470" y2="358"
        :class="['d-arrow', { active: step >= 4 }]"
        stroke="#a855f7" stroke-width="2" marker-end="url(#d-arr-purple)"
      />
      <circle v-if="step === 4" r="5" fill="#a855f7" filter="url(#d-gl-purple)" class="d-particle">
        <animateMotion dur="0.6s" repeatCount="indefinite" path="M425,358 L470,358" />
      </circle>

      <!-- ===== Dream phase nodes ===== -->
      <!-- 1. Prompts (step 4) -->
      <g :class="['d-node', 'phase-node', { active: step >= 4, current: step === 4 }]">
        <rect x="490" y="335" width="100" height="48" rx="4" fill="rgba(168,85,247,0.06)" :stroke="step === 4 ? '#a855f7' : 'rgba(168,85,247,0.4)'" stroke-width="1.5" :filter="step === 4 ? 'url(#d-gl-purple)' : ''" />
        <text x="540" y="355" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">PROMPTS</text>
        <text x="540" y="374" text-anchor="middle" fill="rgba(168,85,247,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step === 4 ? 'updating...' : (step > 4 ? '✓' : '') }}
        </text>
      </g>

      <!-- Arrow: prompts → adapters -->
      <line x1="590" y1="359" x2="615" y2="359" :class="['d-arrow', 'phase-arrow', { active: step >= 5 }]" stroke="#a855f7" stroke-width="1.5" marker-end="url(#d-arr-purple)" />

      <!-- 2. Adapters (step 5) -->
      <g :class="['d-node', 'phase-node', { active: step >= 5, current: step === 5 }]">
        <rect x="618" y="335" width="100" height="48" rx="4" fill="rgba(168,85,247,0.06)" :stroke="step === 5 ? '#a855f7' : 'rgba(168,85,247,0.4)'" stroke-width="1.5" :filter="step === 5 ? 'url(#d-gl-purple)' : ''" />
        <text x="668" y="355" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">ADAPTERS</text>
        <text x="668" y="374" text-anchor="middle" fill="rgba(168,85,247,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step === 5 ? 'tuning...' : (step > 5 ? '✓' : '') }}
        </text>
      </g>

      <!-- Arrow: adapters → evaluation -->
      <line x1="718" y1="359" x2="743" y2="359" :class="['d-arrow', 'phase-arrow', { active: step >= 6 }]" stroke="#a855f7" stroke-width="1.5" marker-end="url(#d-arr-purple)" />

      <!-- 3. Evaluation (step 6) -->
      <g :class="['d-node', 'phase-node', { active: step >= 6, current: step === 6 }]">
        <rect x="746" y="335" width="100" height="48" rx="4" fill="rgba(168,85,247,0.06)" :stroke="step === 6 ? '#a855f7' : 'rgba(168,85,247,0.4)'" stroke-width="1.5" :filter="step === 6 ? 'url(#d-gl-purple)' : ''" />
        <text x="796" y="355" text-anchor="middle" fill="#a855f7" font-family="'Orbitron',sans-serif" font-size="10" font-weight="700" letter-spacing="0.06em">EVALUATE</text>
        <text x="796" y="374" text-anchor="middle" fill="rgba(168,85,247,0.5)" font-family="'Share Tech Mono',monospace" font-size="12">
          {{ step === 6 ? 'scoring...' : (step > 6 ? '96% ✓' : '') }}
        </text>
      </g>

      <!-- Arrow: evaluation → contribute -->
      <line x1="846" y1="359" x2="871" y2="359" :class="['d-arrow', 'phase-arrow', { active: step >= 7 }]" stroke="#a855f7" stroke-width="1.5" marker-end="url(#d-arr-purple)" />

      <!-- 4. Contribute (step 7) -->
      <g :class="['d-node', 'phase-node', { active: step >= 7, current: step === 7 }]">
        <rect x="874" y="335" width="80" height="48" rx="4" fill="rgba(168,85,247,0.06)" :stroke="step === 7 ? '#00ff88' : 'rgba(168,85,247,0.4)'" stroke-width="1.5" :filter="step === 7 ? 'url(#d-gl-green)' : ''" />
        <text x="914" y="353" text-anchor="middle" :fill="step === 7 ? '#00ff88' : '#a855f7'" font-family="'Orbitron',sans-serif" font-size="9" font-weight="700" letter-spacing="0.04em">CONTRIBUTE</text>
        <text x="914" y="372" text-anchor="middle" :fill="step === 7 ? 'rgba(0,255,136,0.6)' : 'rgba(168,85,247,0.5)'" font-family="'Share Tech Mono',monospace" font-size="11">
          {{ step === 7 ? '→ community' : '' }}
        </text>
      </g>

      <!-- Arrow: contribute → community (step 7) -->
      <path
        d="M 914,335 L 914,260 Q 914,220 874,220 L 620,220"
        fill="none"
        :class="['d-arrow', 'd-return', { active: step >= 7 }]"
        stroke="#00ff88" stroke-width="2" marker-end="url(#d-arr-green)"
      />
      <circle v-if="step === 7" r="5" fill="#00ff88" filter="url(#d-gl-green)" class="d-particle">
        <animateMotion dur="1.2s" repeatCount="indefinite" path="M914,335 L914,260 Q914,220 874,220 L620,220" />
      </circle>

      <!-- ===== Model badge ===== -->
      <g :class="['d-node', { active: step >= 5 }]">
        <rect x="590" y="400" width="200" height="34" rx="4" fill="rgba(250,204,21,0.05)" stroke="rgba(250,204,21,0.3)" stroke-width="1" />
        <text x="690" y="422" text-anchor="middle" fill="rgba(250,204,21,0.7)" font-family="'Share Tech Mono',monospace" font-size="12">target: gemini-2.0-flash</text>
      </g>

      <!-- ===== Step progress dots ===== -->
      <g transform="translate(410, 520)">
        <circle
          v-for="(s, i) in steps" :key="'dot-' + i"
          :cx="i * 22" cy="0" r="4"
          :fill="i === step ? '#a855f7' : (i < step ? 'rgba(168,85,247,0.4)' : 'rgba(168,85,247,0.1)')"
          :class="{ 'dot-active': i === step }"
        />
      </g>
    </svg>
  </div>
</template>

<style scoped>
.dream-wrapper {
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
  color: rgba(168, 85, 247, 0.8);
  letter-spacing: 0.05em;
  animation: d-text-appear 0.4s ease forwards;
}

@keyframes d-text-appear {
  0% { opacity: 0; transform: translateY(6px); filter: blur(4px); }
  100% { opacity: 1; transform: translateY(0); filter: blur(0); }
}

.dream-diagram {
  width: 100%;
  height: auto;
  opacity: 0;
  transition: opacity 0.8s ease;
}

.dream-diagram.visible {
  opacity: 1;
}

.d-node {
  opacity: 0.15;
  transition: opacity 0.5s ease;
}

.d-node.active {
  opacity: 1;
}

.phase-node.current rect {
  animation: phase-pulse 1.5s ease-in-out infinite;
}

@keyframes phase-pulse {
  0%, 100% { stroke-opacity: 1; }
  50% { stroke-opacity: 0.4; }
}

.stale {
  opacity: 0.3;
  transition: opacity 0.5s;
}

.d-arrow {
  opacity: 0;
  stroke-dasharray: 400;
  stroke-dashoffset: 400;
  transition: opacity 0.4s ease, stroke-dashoffset 0.6s ease;
}

.d-arrow.active {
  opacity: 0.6;
  stroke-dashoffset: 0;
}

.d-return.active {
  stroke-dasharray: 8 5;
  stroke-dashoffset: 0;
  animation: d-dash-flow 0.8s linear infinite;
}

.phase-arrow.active {
  opacity: 0.4;
}

@keyframes d-dash-flow {
  to { stroke-dashoffset: -26; }
}

.dream-aura {
  animation: aura-breathe 4s ease-in-out infinite;
}

@keyframes aura-breathe {
  0%, 100% { r: 68; stroke-opacity: 0.15; }
  50% { r: 74; stroke-opacity: 0.3; }
}

.spinning {
  transform-origin: 110px 220px;
  animation: d-spin 12s linear infinite;
}

@keyframes d-spin {
  to { transform: rotate(360deg); }
}

.dream-box {
  opacity: 0.1;
  transition: opacity 0.6s, stroke 0.6s;
}

.dream-box.active {
  opacity: 1;
  animation: box-breathe 3s ease-in-out infinite;
}

@keyframes box-breathe {
  0%, 100% { stroke-opacity: 1; }
  50% { stroke-opacity: 0.4; }
}

.d-particle {
  opacity: 0.9;
}

.dot-active {
  animation: d-dot-pulse 1s ease-in-out infinite;
}

@keyframes d-dot-pulse {
  0%, 100% { r: 4; }
  50% { r: 6; }
}

@media (max-width: 640px) {
  .dream-wrapper {
    margin: 1rem auto 0;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .dream-diagram {
    min-width: 600px;
  }
  .step-text {
    font-size: 0.85rem;
  }
}
</style>
