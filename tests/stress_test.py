"""
Stress test harness — sends tasks to the live Sentient brain via Redis.
Monitors brain:response for replies. Flags tasks that get stuck (timeout).
Does NOT fix anything — just observes and logs.

Usage:
    python tests/stress_test.py                      # run default (EML) cases
    python tests/stress_test.py --suite 1000         # run the 1000-case suite
    python tests/stress_test.py --suite 1000 --start 20 --end 50
    python tests/stress_test.py --suite 1000 --tag greeting
    python tests/stress_test.py --only 5,12,42       # run specific cases
    python tests/stress_test.py --timeout 90         # custom timeout per task
    python tests/stress_test.py --tag eml-1          # run only stage 1 cases
"""

import argparse
import json
import os
import sys
import threading
import time

import redis

# Force unbuffered output so progress is visible when piped/redirected
os.environ.setdefault("PYTHONUNBUFFERED", "1")

WORKDIR = os.path.expanduser("~/Documents/projects")
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "html_to_eml", "fixtures")
TIMEOUT_DEFAULT = 180  # seconds per task (longer for code nerves)

# ── HTML-to-EML Pipeline Cases ─────────────────────────────────────────────
# Sequential: each stage builds on the previous one's output.
# The brain synthesizes nerves as it encounters new task domains.
# Run with: --tag eml-1 through eml-11 for specific stages.

CASES = [
    # ── Stage 1: HTML Scraping (1-5) ──
    # Nerve: html_scraper_nerve — fetch raw HTML+CSS from URLs or local files
    {"id": 1, "tag": "eml-1", "msgs": [
        "Fetch the complete raw HTML source code from https://example.com — I need the full markup with all tags, classes, attributes, and inline styles preserved. Do NOT extract just the text, return the actual HTML source."
    ]},
    {"id": 2, "tag": "eml-1", "msgs": [
        "Get the raw HTML and all CSS rules (from <style> tags and linked stylesheets) from https://example.com. Return both the HTML markup and the CSS content."
    ]},
    {"id": 3, "tag": "eml-1", "msgs": [
        f"Read the HTML file at {FIXTURE_DIR}/01_raw_html/sample_page.html and tell me how many top-level sections it has. List each section's tag name and id attribute."
    ]},
    {"id": 4, "tag": "eml-1", "msgs": [
        "Fetch the full HTML from https://httpbin.org/html preserving all element classes, IDs, and data attributes intact. Return the complete DOM markup, not a text summary."
    ]},
    {"id": 5, "tag": "eml-1", "msgs": [
        f"Read {FIXTURE_DIR}/01_raw_html/sample_page.html and extract all the CSS — both the inline <style> block and any style= attributes on elements. Return the CSS as a single text block."
    ]},

    # ── Stage 2: Platform Detection (6-10) ──
    # Nerve: platform_detector_nerve — identify website builder from HTML patterns
    {"id": 6, "tag": "eml-2", "msgs": [
        "Detect the website platform from this HTML: <div class='w-section'><div class='w-container'><div class='w-row' data-wf-page='abc123'><div class='w-col w-col-6'>Content</div></div></div></div>. Tell me what platform built this and what signals you found."
    ]},
    {"id": 7, "tag": "eml-2", "msgs": [
        "What platform built this HTML: <div class='framer-abc123' data-framer-component-type='RichTextContainer'><div class='framer-xyz789' style='--framer-text-color: red'>Text</div></div>? List the specific class patterns and data attributes that identify it."
    ]},
    {"id": 8, "tag": "eml-2", "msgs": [
        "Identify the platform from this HTML: <div class='elementor-section elementor-top-section' data-element_type='section'><div class='elementor-container elementor-column-gap-default'><div class='elementor-widget elementor-widget-heading'>Title</div></div></div>"
    ]},
    {"id": 9, "tag": "eml-2", "msgs": [
        f"Read the HTML at {FIXTURE_DIR}/01_raw_html/sample_page.html and detect what website platform was used to build it. Output JSON with: platform name, confidence score, and the specific signals found (class patterns, data attributes, meta tags)."
    ]},
    {"id": 10, "tag": "eml-2", "msgs": [
        "Detect the platform from: <header class='site-header'><nav class='main-nav'><ul><li><a href='/'>Home</a></li></ul></nav></header><section><div class='container'><h1>Hello</h1></div></section>. No framework-specific classes — what does that tell you?"
    ]},

    # ── Stage 3: Theme Extraction for Wix EML (11-16) ──
    # Nerve: wix_theme_extractor_nerve — extract colors/fonts and map to Wix --wst-* variables
    # Should discover tools: eml_theme_variables, eml_map_colors_to_theme, eml_resolve_font, eml_font_list
    {"id": 11, "tag": "eml-3", "msgs": [
        "Extract the color theme from this CSS and map each color to its Wix EML --wst-* theme variable. CSS: body { background-color: #ffffff; color: #333333; } h1, h2 { color: #1a1a2e; } .btn-primary { background: #e94560; } a { color: #0f3460; } .border { border-color: #e0e0e0; }. I need: --wst-base-1-color, --wst-base-2-color, --wst-shade-1-color, --wst-accent-1-color, etc. Return the complete Wix theme mapping as JSON."
    ]},
    {"id": 12, "tag": "eml-3", "msgs": [
        "Extract fonts from this CSS and resolve each to a Wix-supported font: body { font-family: 'Inter', sans-serif; } h1, h2 { font-family: 'Playfair Display', serif; font-weight: 700; }. Map to --wst-heading-1-font through heading-6 and --wst-paragraph-1-font through paragraph-3. List available Wix fonts if the exact font isn't supported."
    ]},
    {"id": 13, "tag": "eml-3", "msgs": [
        "Extract CSS variables from: :root { --primary: #2563eb; --secondary: #7c3aed; --bg: #f8fafc; --text: #1e293b; } and convert them to Wix Harmony theme variables (--wst-base-1-color, --wst-accent-1-color, etc). Also derive the shade variables (shade-1 through shade-3) using the color mixing formula."
    ]},
    {"id": 14, "tag": "eml-3", "msgs": [
        f"Read {FIXTURE_DIR}/01_raw_html/sample_page.html and extract the complete Wix EML theme: map all colors to --wst-* color variables (base, shade, accent), resolve all fonts to Wix-available fonts, and build the full WixThemeConfig with fontVariables (heading-1..6, paragraph-1..3). Output as JSON."
    ]},
    {"id": 15, "tag": "eml-3", "msgs": [
        "Extract theme from CSS with complex colors for Wix EML mapping: .hero { background: rgba(255, 107, 107, 0.9); } p { color: hsl(220, 13%, 18%); } .cta { background: linear-gradient(90deg, #667eea, #764ba2); }. Convert all formats to hex, extract dominant color from gradients, then map to --wst-* variables."
    ]},
    {"id": 16, "tag": "eml-3", "msgs": [
        "I have these site colors: background=#ffffff, text=#1e293b, heading=#0f172a, accent=#2563eb, accentAlt=#7c3aed, border=#e2e8f0, backgroundAlt=#f8fafc. Map them to the complete Wix EML theme: --wst-base-1-color through --wst-accent-3-color. Derive shade-2 as the midpoint between text and background."
    ]},

    # ── Stage 4: Section Splitting (17-22) ──
    # Nerve: html_section_splitter_nerve — split HTML into individual sections for EML conversion
    {"id": 17, "tag": "eml-4", "msgs": [
        f"Read {FIXTURE_DIR}/01_raw_html/sample_page.html. Split it into individual sections for Wix EML conversion. Each section will become a separate EML <Section> (or <Header>/<Footer>). Identify type (header, hero, features, testimonials, cta, footer), mark header/footer as shared, and output a sections.json index."
    ]},
    {"id": 18, "tag": "eml-4", "msgs": [
        "Split this HTML into sections for EML conversion: <header><nav>Logo Menu</nav></header><section class='hero'><h1>Welcome</h1><p>Tagline</p></section><section class='features'><h2>Features</h2><div>Feature 1</div></section><footer><p>Copyright 2024</p></footer>. Note: header maps to EML <Header> tag, footer to EML <Footer> tag, others to <Section>."
    ]},
    {"id": 19, "tag": "eml-4", "msgs": [
        f"Read {FIXTURE_DIR}/01_raw_html/sample_page.html. The header has position:sticky — flag it as fixed. Mark header and footer as shared components (reused across pages in Wix). Output sections.json with is_fixed, is_shared, and eml_root_tag (Header/Footer/Section) fields."
    ]},
    {"id": 20, "tag": "eml-4", "msgs": [
        "Split deeply nested HTML into EML sections: <body><div class='page-wrapper'><div class='content-wrapper'><header><nav>Menu</nav></header><div class='main-content'><section><h1>Hero</h1></section><section><h2>About</h2></section></div><footer>Footer</footer></div></div></body>. Unwrap structural wrappers and find actual content sections."
    ]},
    {"id": 21, "tag": "eml-4", "msgs": [
        "Split HTML with no semantic tags into EML sections: <body><div id='nav'><a>Logo</a><a>Link</a></div><div id='main'><div id='block1'><h1>Title</h1></div><div id='block2'><h2>Features</h2></div></div><div id='foot'><p>Copyright</p></div></body>. Classify by content to determine which EML root tag each gets."
    ]},
    {"id": 22, "tag": "eml-4", "msgs": [
        "Split a single-section page for EML: <body><div class='page'><h1>One Page</h1><p>Just one block of content.</p></div></body>. This becomes one EML <Section>. Output sections.json with the single entry."
    ]},

    # ── Stage 5: Component Tree for Wix EML (23-28) ──
    # Nerve: eml_component_tree_nerve — classify HTML elements as Wix EML component types
    # Should discover tools: eml_component_spec, eml_list_components
    {"id": 23, "tag": "eml-5", "msgs": [
        "Build a Wix EML component tree from this hero HTML. Classify each element as a Wix EML type (Section, Container, Text, Image, Button, Line). HTML: <section id='hero' style='background:#0f172a;padding:120px 24px;text-align:center'><div style='max-width:800px;margin:0 auto'><h1 style='font-size:56px;color:white'>Build Better Products</h1><p style='font-size:20px;color:#94a3b8'>The all-in-one platform.</p><div style='display:flex;gap:16px;justify-content:center'><a style='padding:16px 32px;background:#2563eb;color:white;border-radius:8px'>Start Free</a><a style='padding:16px 32px;border:2px solid #475569;color:white'>Watch Demo</a></div></div></section>. Use eml_component_spec to verify valid types. Output JSON tree."
    ]},
    {"id": 24, "tag": "eml-5", "msgs": [
        "Build a Wix EML component tree from a 3-column features grid. Each card has icon+title+description. The grid container maps to Container with grid layout, each card to Container, <img> to Image, <h3> to Text, <p> to Text. HTML: <section style='padding:100px;background:#f8fafc'><h2 style='text-align:center'>Features</h2><div style='display:grid;grid-template-columns:repeat(3,1fr);gap:32px'><div><img src='i.svg' width='48'><h3>Fast</h3><p>Speed.</p></div><div><img src='i2.svg' width='48'><h3>Secure</h3><p>Safe.</p></div><div><img src='i3.svg' width='48'><h3>Easy</h3><p>Simple.</p></div></div></section>. Output JSON."
    ]},
    {"id": 25, "tag": "eml-5", "msgs": [
        "Build a Wix EML component tree for a navigation header. In EML, this uses the Header component (NOT Section). Map: <img> logo to Image or Logo, nav links to Menu or Text, CTA <a> to Button. HTML: <header><nav style='display:flex;justify-content:space-between;align-items:center;padding:16px'><img src='logo.svg' height='32'><ul style='display:flex;gap:32px'><li><a>Features</a></li><li><a>Pricing</a></li></ul><a style='padding:10px 24px;background:#2563eb;color:white;border-radius:8px'>Get Started</a></nav></header>"
    ]},
    {"id": 26, "tag": "eml-5", "msgs": [
        "Build a Wix EML component tree for a testimonial card. Map quote text to Text with richText, avatar to Image (displayMode:fill), name to bold Text, role to muted Text. HTML: <div style='padding:32px;background:#f8fafc'><p>\"Great product!\"</p><div style='display:flex;align-items:center;gap:12px'><img src='avatar.jpg' width='44' style='border-radius:50%'><div><p style='font-weight:600'>Jane Doe</p><p style='color:#64748b;font-size:13px'>CEO, Company</p></div></div></div>"
    ]},
    {"id": 27, "tag": "eml-5", "msgs": [
        "Build Wix EML component tree with an <hr> element. In EML, <hr> maps to the Line component. HTML: <section><div style='text-align:center'><h2>Title</h2><hr style='width:80px;border:2px solid #e94560;margin:20px auto'><p>Content below</p></div></section>. Check the Line component spec for valid preset and cssCustomProperties."
    ]},
    {"id": 28, "tag": "eml-5", "msgs": [
        f"Read the component tree at {FIXTURE_DIR}/05_component_trees/01-hero.json and validate it against Wix EML rules: Section must be root, children must be valid EML types (Container, Text, Image, Button, Line, VectorArt). Check each component's spec for required fields."
    ]},

    # ── Stage 6: CSS to Wix EML Tailwind (29-35) ──
    # Nerve: eml_css_to_tailwind_nerve — convert CSS to Wix EML-compatible Tailwind classes
    # Should discover tools: eml_css_to_tailwind, eml_tailwind_rules
    {"id": 29, "tag": "eml-6", "msgs": [
        "Convert these CSS properties to Wix EML-compatible Tailwind classes. IMPORTANT: Wix EML has specific Tailwind restrictions (no decimal gaps, no bracket gaps, typography goes in cssProperties not classes). CSS: display:flex; flex-direction:column; justify-content:center; align-items:center; gap:20px; padding:40px;. Check the EML Tailwind rules for what's allowed."
    ]},
    {"id": 30, "tag": "eml-6", "msgs": [
        "Convert to Wix EML Tailwind: width:100%; max-width:1200px; margin:0 auto; padding:60px 24px; background-color:#f8fafc;. Note: in Wix EML, background-color goes in cssCustomProperties, not Tailwind classes. Separate layout classes from style properties."
    ]},
    {"id": 31, "tag": "eml-6", "msgs": [
        "Convert these TEXT styles for Wix EML: font-size:56px; font-weight:800; line-height:1.1; color:white; letter-spacing:-0.02em;. In Wix EML, typography MUST go in cssProperties (not Tailwind classes). Return: which go to cssProperties vs Tailwind classes."
    ]},
    {"id": 32, "tag": "eml-6", "msgs": [
        "Convert a 3-column grid for Wix EML Tailwind: display:grid; grid-template-columns:repeat(3,1fr); gap:32px;. In Wix EML, grid columns use grid-cols-[1fr_1fr_1fr] format (underscore separator). What about gap — is gap-8 allowed or must it be gap-0?"
    ]},
    {"id": 33, "tag": "eml-6", "msgs": [
        "Convert button styles for Wix EML: padding:16px 32px; background:#2563eb; color:white; border-radius:8px; font-weight:600;. In EML Buttons, label styling goes in elements.label.cssProperties, background in cssProperties, not Tailwind classes. Split correctly."
    ]},
    {"id": 34, "tag": "eml-6", "msgs": [
        "Convert child positioning for Wix EML: element at x=100, y=50, width=600 inside parent width=1200. Wix EML uses bounding-box positioning: ml-[N%], mt-[Npx], w-[N%]. Calculate: ml-[8.33%] mt-[50px] w-[50%]. Verify against EML Tailwind rules."
    ]},
    {"id": 35, "tag": "eml-6", "msgs": [
        "Convert edge-case CSS to Wix EML Tailwind: gap:1.5px; position:sticky; z-index:100;. Wix EML forbids decimal gaps (gap-1.5 is invalid — use gap-1 or gap-2). Position:sticky and z-index are NOT supported in EML Tailwind. What are the correct alternatives?"
    ]},

    # ── Stage 7: Wix Theme Variable Mapping (36-41) ──
    # Nerve: wix_theme_mapper_nerve — map extracted colors/fonts to --wst-* variables
    # Should discover tools: eml_theme_variables, eml_map_colors_to_theme, eml_resolve_font
    {"id": 36, "tag": "eml-7", "msgs": [
        "Map these extracted site colors to Wix Harmony --wst-* theme variables: background=#ffffff, text=#333333, heading=#1a1a2e, accent=#e94560, border=#e0e0e0. I need the full mapping including derived shades (shade-2 = midpoint of text+bg). Use the Wix EML theme variable spec to get the correct variable names."
    ]},
    {"id": 37, "tag": "eml-7", "msgs": [
        "Map fonts to Wix theme: heading font 'Montserrat' 700, body font 'Inter' 400. First check if these fonts are available in Wix. Then map to --wst-heading-1-font format: 'normal normal 700 48px/1.2em Montserrat'. Build fontVariables for heading-1..6 and paragraph-1..3."
    ]},
    {"id": 38, "tag": "eml-7", "msgs": [
        "Map a dark theme to Wix --wst-* variables: bg=#0f172a, text=#e0e0e0, heading=#ffffff, accent=#2563eb, accentAlt=#7c3aed, border=#334155. Dark themes swap base-1/base-2 roles. Derive shade-2 and shade-3 using the color mixing formula."
    ]},
    {"id": 39, "tag": "eml-7", "msgs": [
        f"Read the extracted theme at {FIXTURE_DIR}/03_theme/sample_page.json. Map all colors to --wst-* variables and resolve all fonts to Wix-available fonts. Output the complete WixThemeConfig JSON with colors, fonts, and fontVariables."
    ]},
    {"id": 40, "tag": "eml-7", "msgs": [
        "I have 6 accent colors but Wix only supports --wst-accent-1 through accent-3. Colors by importance: primary=#e94560, secondary=#2563eb, tertiary=#7c3aed, success=#059669, warning=#d97706, error=#dc2626. Map top 3 to accent vars, keep the rest as hardcoded hex in component cssProperties."
    ]},
    {"id": 41, "tag": "eml-7", "msgs": [
        "Resolve the font 'Poppins' for Wix EML. Is it available in Wix? If so, which variant (Regular, Bold, Semi Bold)? Build the --wst-heading-1-font value in the correct format: 'normal normal WEIGHT SIZEpx/LINE_HEIGHT FAMILY'."
    ]},

    # ── Stage 8: EML JSX Generation (42-49) ──
    # Nerve: eml_generator_nerve — generate valid Wix EML JSX from component trees
    # Should discover tools: eml_component_spec, eml_layout_pattern, eml_richtext_format, eml_background_rules
    {"id": 42, "tag": "eml-8", "msgs": [
        "Generate valid Wix EML JSX for a hero section. Use the flex-column-centered layout pattern. Requirements: Section with Background element (var(--wst-shade-1-color)), Container child, Text h1 'Build Better Products' with data.richText (must include type:'Builder.RichText'), Text subtitle, two Buttons with preset='baseButton'. Check the EML component specs for required attributes on each type."
    ]},
    {"id": 43, "tag": "eml-8", "msgs": [
        "Generate Wix EML JSX for a 3-column features grid. Use the flex-row-three-column layout pattern. Section bg var(--wst-shade-3-color), Container with grid grid-cols-[1fr_1fr_1fr], 3 card Containers each with Image (data.image.type must be 'Builder.Image', displayMode:'fit') and Text elements. Follow EML validation rules for all attributes."
    ]},
    {"id": 44, "tag": "eml-8", "msgs": [
        "Generate Wix EML JSX for a site header. CRITICAL: use <Header> root tag, NOT <Section>. In Header, richText MUST include type:'Builder.RichText'. Use spx font units (not px). Layout: grid with percentage-based ml-[X%] positioning for logo, nav links, and CTA button. Check the Header component spec for the exact rules."
    ]},
    {"id": 45, "tag": "eml-8", "msgs": [
        "Generate Wix EML JSX for a site footer. CRITICAL: use <Footer> root tag, NOT <Section>. In Footer, richText must NOT include the type field (opposite of Header!). Dark background var(--wst-base-2-color), light text var(--wst-base-1-color). Use spx font units. Check the Footer component spec."
    ]},
    {"id": 46, "tag": "eml-8", "msgs": [
        "Generate Wix EML JSX for a testimonial card. Container with Background element (cssCustomProperties.backgroundColor), Text for quote (richText with type:'Builder.RichText'), flex Container for avatar row: Image (44x44, displayMode:'fill', image.type:'Builder.Image'), Text name (bold), Text role (muted color). All theme colors via --wst-* vars."
    ]},
    {"id": 47, "tag": "eml-8", "msgs": [
        "Generate Wix EML JSX for a CTA section. Section with dark Background, centered Text h2, Text subtitle, and a Button with preset='baseButton'. EML doesn't support Form — use Container + Button for form-like layouts. Check which cssProperties are allowed on Button vs elements.label.cssProperties."
    ]},
    {"id": 48, "tag": "eml-8", "msgs": [
        f"Read the component tree at {FIXTURE_DIR}/05_component_trees/01-hero.json and the Tailwind mappings at {FIXTURE_DIR}/06_tailwind/01-hero.json. Generate EML JSX from these inputs. Use eml_component_spec to verify each component's required attributes. Compare your output to {FIXTURE_DIR}/08_eml/01-hero.eml.jsx."
    ]},
    {"id": 49, "tag": "eml-8", "msgs": [
        "Generate the minimum valid Wix EML JSX for an empty section. Check the Section component spec for required attributes: id, classes (h-auto min-h-0 flex flex-col), elements.Background with cssCustomProperties. What is the smallest valid EML that will pass validation?"
    ]},

    # ── Stage 9: EML Validation (50-55) ──
    # Nerve: eml_validator_nerve — validate EML JSX against Wix rules
    # Should discover tools: eml_validation_rules, eml_manifest_filters, eml_richtext_format, eml_tailwind_rules
    {"id": 50, "tag": "eml-9", "msgs": [
        f"Validate the EML file at {FIXTURE_DIR}/08_eml/01-hero.eml.jsx against the full Wix EML validation checklist: JSX syntax, component hierarchy, required attributes, Tailwind class validity, cssProperties placement (typography only on Text/Button), theme variable format, richText structure, image data structure. Output a pass/fail report for each category."
    ]},
    {"id": 51, "tag": "eml-9", "msgs": [
        "Validate this Wix EML and list ALL errors: <Section><Container classes='flex gap-1.5'><Text data={{richText:{text:'<p>Hello</p>'}}}/></Container></Section>. Check: gap-1.5 (decimal not allowed in EML Tailwind), missing Section id, missing Background element, missing richText.type field, missing Container id. Use the EML validation rules."
    ]},
    {"id": 52, "tag": "eml-9", "msgs": [
        "Validate this Image in Wix EML: <Image id='img1' data={{image:{uri:'https://example.com/photo.jpg',width:800,height:600}}}/>. Check against EML image data rules: missing image.type:'Builder.Image', external URL instead of wix:image:// URI, missing displayMode. What are all the issues?"
    ]},
    {"id": 53, "tag": "eml-9", "msgs": [
        "Validate this Text in Wix EML: <Text classes='text-[36px] font-bold' data={{richText:{type:'Builder.RichText',text:'<h1>Title</h1>'}}}/>. Check the EML manifest filter rules: text-[36px] and font-bold are typography — they MUST go in cssProperties, NOT Tailwind classes. What's the corrected version?"
    ]},
    {"id": 54, "tag": "eml-9", "msgs": [
        "Validate this Footer Text in Wix EML: <Footer id='f1'><Text id='t1' data={{richText:{type:'Builder.RichText',text:'<p>Copyright</p>',linkList:[]}}}/></Footer>. CRITICAL Footer rule: richText must NOT include the type field. Is this valid? What needs to change?"
    ]},
    {"id": 55, "tag": "eml-9", "msgs": [
        "Validate this Section for cssProperties placement: <Section id='s1' classes='h-auto' cssProperties={{color:'red',fontSize:'16px',backgroundColor:'blue'}}><Text id='t1'/></Section>. Check manifest filters: Section cannot have typography cssProperties (color, fontSize). backgroundColor goes in elements.Background.cssCustomProperties, not Section cssProperties."
    ]},

    # ── Stage 10: File Organization (56-60) ──
    # Nerve: eml_file_organizer_nerve — create output directory structure for EML conversion
    {"id": 56, "tag": "eml-10", "msgs": [
        "Create an EML output folder at sandbox/eml-output/acme-saas-com for a Wix EML conversion. Site has 1 page (home), 6 sections (header, hero, features, testimonials, cta, footer). Header/footer are shared. Create: pages/home/sections/00-header.eml.jsx through 05-footer.eml.jsx, shared/headers/, shared/footers/, and website-metadata.json."
    ]},
    {"id": 57, "tag": "eml-10", "msgs": [
        "Create sections.json at sandbox/eml-output/acme-saas-com/pages/home/ listing 6 EML sections with: index (00-05), section_type, eml_root_tag (Header/Section/Footer), is_shared flag, and file_path to the .eml.jsx file."
    ]},
    {"id": 58, "tag": "eml-10", "msgs": [
        "Write website-metadata.json at sandbox/eml-output/acme-saas-com/ with: platform, wix_theme (--wst-* color and font variable mappings), pages array, sections_count, and extracted colorRoles/fontRoles."
    ]},
    {"id": 59, "tag": "eml-10", "msgs": [
        f"Read {FIXTURE_DIR}/10_output/expected_structure.json and create this exact EML output directory structure under sandbox/eml-output/. Create all directories and placeholder .eml.jsx files."
    ]},
    {"id": 60, "tag": "eml-10", "msgs": [
        "Create EML output structure for URL https://my-site.co.uk/landing-page?utm_source=google. Sanitize domain to valid dir name (my-site-co-uk), strip query params, create standard Wix EML output layout under sandbox/eml-output/."
    ]},

    # ── Stage 11: Forced Tool Discovery (61-72) ──
    # These cases explicitly name MCP tools so the brain MUST discover and wire them.
    # The brain passes the task as trigger_task -> nerve description -> keyword match -> tool binding.
    {"id": 61, "tag": "eml-11", "msgs": [
        "Use the eml_component_spec tool to look up the Section component spec for Wix EML. Then generate a minimal valid Section with the correct required attributes (id, classes, elements.Background). Return the JSX."
    ]},
    {"id": 62, "tag": "eml-11", "msgs": [
        "Use the eml_list_components tool to get all available Wix EML component types. Then pick the ones needed for a hero section (Section, Container, Text, Image, Button) and build a component tree JSON."
    ]},
    {"id": 63, "tag": "eml-11", "msgs": [
        "Use the eml_theme_variables tool to get the full list of Wix --wst-* theme variables. Then use eml_map_colors_to_theme to map these colors: background=#ffffff, text=#1e293b, accent=#2563eb. Return the complete theme JSON."
    ]},
    {"id": 64, "tag": "eml-11", "msgs": [
        "Use the eml_resolve_font tool to check if 'Montserrat' is available in Wix. Then use eml_font_list to see all available fonts. Build the --wst-heading-1-font value for Montserrat Bold."
    ]},
    {"id": 65, "tag": "eml-11", "msgs": [
        "Use the eml_tailwind_rules tool to get the Wix EML Tailwind restrictions. Then use eml_css_to_tailwind to convert: display:grid; grid-template-columns:repeat(3,1fr); gap:32px; padding:60px;. Return the valid EML classes."
    ]},
    {"id": 66, "tag": "eml-11", "msgs": [
        "Use the eml_layout_pattern tool to get the 'flex-column-centered' layout pattern for Wix EML. Adapt it for a pricing section with 3 pricing cards. Return the JSX."
    ]},
    {"id": 67, "tag": "eml-11", "msgs": [
        "Use the eml_validation_rules tool to get the full Wix EML validation checklist. Then validate this EML: <Section id='s1'><Text classes='text-xl font-bold' data={{richText:{text:'<p>Hello</p>'}}}/></Section>. List every violation found."
    ]},
    {"id": 68, "tag": "eml-11", "msgs": [
        "Use the eml_richtext_format tool to get the richText specification for Wix EML. Then create a valid richText object for a heading that says 'Welcome to Our Site' with font-size 48px and color #1e293b. Return the data.richText JSON."
    ]},
    {"id": 69, "tag": "eml-11", "msgs": [
        "Use the eml_background_rules tool to learn how Section backgrounds work in Wix EML. Create a Section with a dark gradient-like background (use dominant color extraction: linear-gradient(90deg, #667eea, #764ba2) -> dominant hex). Return the Background element JSON."
    ]},
    {"id": 70, "tag": "eml-11", "msgs": [
        "Use the eml_animation_spec tool to get all 'entrance' animations available in Wix EML. Pick FadeIn and SlideIn, then use eml_animation_rules to check if they can be combined. Return the correct entranceAnimation prop JSX."
    ]},
    {"id": 71, "tag": "eml-11", "msgs": [
        "Use the eml_manifest_filters tool to check which CSS properties are allowed on which Wix EML component types. A Section has cssProperties={{color:'red', fontSize:'16px'}} — is that valid? What about a Text component? Fix any violations."
    ]},
    {"id": 72, "tag": "eml-11", "msgs": [
        "Use eml_component_spec to look up the Header component. Then use eml_component_spec again for Footer. What is the critical difference in how richText.type is handled between Header and Footer? Generate a valid Header Text and a valid Footer Text."
    ]},

    # ── Stage 12: Full Pipeline (73-79) ──
    # Nerve: eml_pipeline_nerve — orchestrate full HTML-to-Wix-EML conversion
    # Should use all previous nerves + all wix_eml_* tools
    {"id": 73, "tag": "eml-12", "msgs": [
        f"Convert {FIXTURE_DIR}/01_raw_html/sample_page.html to Wix EML format. Full pipeline: 1) detect platform, 2) extract theme and map to --wst-* variables, 3) split into sections, 4) build component trees (classify as Section/Container/Text/Image/Button), 5) convert CSS to EML-compatible Tailwind, 6) generate EML JSX per section with proper richText format and Background elements. Save all output to sandbox/eml-pipeline-output/. Show each step."
    ]},
    {"id": 74, "tag": "eml-12", "msgs": [
        f"Read {FIXTURE_DIR}/01_raw_html/sample_page.html and produce a .eml.jsx file for EVERY section. Use the eml_component_spec tool for Header/Footer/Section rules. Header richText MUST include type:'Builder.RichText', Footer must NOT. Use --wst-* theme vars, EML-valid Tailwind classes. Save to sandbox/eml-full-output/."
    ]},
    {"id": 75, "tag": "eml-12", "msgs": [
        f"Convert just the hero section from {FIXTURE_DIR}/01_raw_html/sample_page.html to Wix EML. Use eml_layout_pattern for 'flex-column-centered', eml_component_spec for each type, eml_tailwind_rules for class validation. Save to sandbox/eml-hero-only/hero.eml.jsx."
    ]},
    {"id": 76, "tag": "eml-12", "msgs": [
        f"Convert the features section from {FIXTURE_DIR}/01_raw_html/sample_page.html to Wix EML. Use eml_layout_pattern for 'flex-row-three-column', eml_component_spec for Image (type:'Builder.Image') and Text specs. Save to sandbox/eml-features/features.eml.jsx."
    ]},
    {"id": 77, "tag": "eml-12", "msgs": [
        "Convert this minimal HTML to valid Wix EML JSX: <html><body><section style='background:#f0f0f0;padding:60px 20px;text-align:center'><h1 style='font-size:48px;color:#111'>Hello World</h1><p style='font-size:18px;color:#666'>Simple page.</p></section></body></html>. Use eml_component_spec for Section, eml_richtext_format for Text data, eml_background_rules for Background. Save to sandbox/eml-minimal/section.eml.jsx."
    ]},
    {"id": 78, "tag": "eml-12", "msgs": [
        f"Full Wix EML package from {FIXTURE_DIR}/01_raw_html/sample_page.html: use eml_theme_variables + eml_map_colors_to_theme for theme, eml_validation_rules to validate each section, eml_component_spec for every component type. Generate website-metadata.json, sections.json, and .eml.jsx per section. Save to sandbox/eml-complete/."
    ]},
    {"id": 79, "tag": "eml-12", "msgs": [
        "Use eml_list_components to get all Wix EML types, eml_list_layouts to get all layout patterns, and eml_animation_rules to get animation options. Given a landing page with: hero (dark bg, centered text, CTA), features (3 columns), testimonials (cards), pricing (3 tiers), and footer — generate one .eml.jsx per section using the correct tools for each. Save to sandbox/eml-landing/."
    ]},
]


STRESS_TEST_USER_ID = "stress-test-user-001"
STRESS_TEST_EMAIL = "stress-test@arqitect.local"


def setup_test_user(r_pub):
    """Create a test user in cold memory so the brain treats us as identified.

    Bypasses onboarding by inserting the user directly into knowledge.db
    and passing user_id in every task message.
    """
    db_path = os.path.join(os.path.dirname(__file__), "..", "knowledge.db")
    if not os.path.exists(db_path):
        print(f"[SETUP] knowledge.db not found at {db_path} — skipping user setup")
        return

    import sqlite3
    conn = sqlite3.connect(db_path)
    # Insert user if not exists
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, display_name) VALUES (?, ?)",
        (STRESS_TEST_USER_ID, "Stress Tester"),
    )
    conn.commit()
    conn.close()
    print(f"[SETUP] Test user ready: {STRESS_TEST_USER_ID}")


def run_case(case, r_pub, r_sub, timeout):
    """Send messages for one case, collect responses. Returns result dict."""
    case_id = case["id"]
    tag = case["tag"]
    msgs = case["msgs"]
    results = []

    for i, msg in enumerate(msgs):
        step = f"[Case {case_id}/{tag}] msg {i+1}/{len(msgs)}"
        print(f"\n{'='*60}")
        print(f"{step}: {msg[:80]}{'...' if len(msg) > 80 else ''}")
        print(f"{'='*60}")

        # Subscribe for response before publishing
        response_event = threading.Event()
        response_data = {"text": None, "elapsed": 0}

        pubsub = r_sub.pubsub()
        pubsub.subscribe("brain:response")

        def listen_for_response():
            start = time.time()
            for raw_msg in pubsub.listen():
                if raw_msg["type"] != "message":
                    continue
                try:
                    data = json.loads(raw_msg["data"])
                    text = data.get("message", "")
                    if text:
                        response_data["text"] = text
                        response_data["elapsed"] = round(time.time() - start, 1)
                        response_event.set()
                        return
                except Exception:
                    pass

        listener = threading.Thread(target=listen_for_response, daemon=True)
        listener.start()

        # Publish task with test user identity
        start_time = time.time()
        r_pub.publish("brain:task", json.dumps({
            "task": msg,
            "source": "stress_test",
            "user_id": STRESS_TEST_USER_ID,
        }))

        # Wait for response or timeout
        got_response = response_event.wait(timeout=timeout)
        pubsub.unsubscribe()
        pubsub.close()

        elapsed = round(time.time() - start_time, 1)

        if got_response:
            resp_text = response_data["text"] or ""
            status = "ok"
            preview = resp_text[:120].replace("\n", " ")
            print(f"  -> [{response_data['elapsed']}s] {preview}{'...' if len(resp_text) > 120 else ''}")
        else:
            status = "STUCK"
            resp_text = ""
            print(f"  -> STUCK (no response after {timeout}s)")

        results.append({
            "msg": msg[:100],
            "status": status,
            "elapsed": elapsed,
            "response": resp_text[:200] if resp_text else "",
        })

        # Brief pause between multi-turn messages
        if i < len(msgs) - 1:
            time.sleep(2)

    return {
        "id": case_id,
        "tag": tag,
        "steps": results,
        "overall": "STUCK" if any(r["status"] == "STUCK" for r in results) else "ok",
    }


def _wait_for_qualification(r_pub, max_wait=15):
    """Wait until all nerves have been qualified (at least 1 iteration with qualified=1).

    Polls the knowledge DB every 5 seconds. Gives up after max_wait seconds.
    This ensures nerves finish their first qualification round before the next task.
    """
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "..", "knowledge.db")
    if not os.path.exists(db_path):
        return

    start = time.time()
    while time.time() - start < max_wait:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            # Find nerves that exist in registry but haven't qualified yet
            c.execute("""
                SELECT nr.name
                FROM nerve_registry nr
                LEFT JOIN qualification_results qr
                  ON qr.subject_name = nr.name AND qr.subject_type = 'nerve'
                WHERE nr.is_sense = 0
                  AND (qr.qualified IS NULL OR qr.qualified = 0)
            """)
            pending = [r["name"] for r in c.fetchall()]
            conn.close()

            if not pending:
                return  # All nerves are qualified

            elapsed = int(time.time() - start)
            print(f"  [WAIT] {len(pending)} nerve(s) still qualifying: {', '.join(pending[:3])}... ({elapsed}s)", flush=True)
            time.sleep(5)
        except Exception as e:
            print(f"  [WAIT] DB check error: {e}", flush=True)
            time.sleep(5)

    print(f"  [WAIT] Timed out after {max_wait}s — proceeding anyway", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Sentient stress test")
    parser.add_argument("--suite", type=str, default="eml", help="Case suite: 'eml' (default 79) or '1000'")
    parser.add_argument("--start", type=int, default=1, help="Start from case N")
    parser.add_argument("--end", type=int, default=None, help="End at case N (default: all)")
    parser.add_argument("--only", type=str, default="", help="Comma-separated case IDs")
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT, help="Seconds to wait per message")
    parser.add_argument("--tag", type=str, default="", help="Run only cases with this tag")
    parser.add_argument("--dream-wait", type=int, default=0,
                        help="Seconds to wait after cases for dreamstate (0=skip, 300=5min)")
    args = parser.parse_args()

    # Load the appropriate case suite
    if args.suite == "1000":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "stress_cases_1000",
            os.path.join(os.path.dirname(__file__), "stress_cases_1000.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        case_list = mod.CASES
    else:
        case_list = CASES

    # Default end to the last case in the suite
    end = args.end if args.end is not None else max(c["id"] for c in case_list)

    r_pub = redis.Redis(host="localhost", port=6379, decode_responses=True)
    r_sub = redis.Redis(host="localhost", port=6379, decode_responses=True)

    # Verify Redis is up
    try:
        r_pub.ping()
    except redis.ConnectionError:
        print("ERROR: Redis not running. Start arqitect first: bash start.sh")
        sys.exit(1)

    # Set up test user so the brain treats us as identified
    setup_test_user(r_pub)

    # Select cases
    if args.only:
        ids = [int(x) for x in args.only.split(",")]
        cases = [c for c in case_list if c["id"] in ids]
    elif args.tag:
        cases = [c for c in case_list if args.tag in c["tag"]]
    else:
        cases = [c for c in case_list if args.start <= c["id"] <= end]

    print(f"\nSentient Stress Test — {len(cases)} cases, timeout={args.timeout}s")
    print(f"{'='*60}\n")

    all_results = []
    ok_count = 0
    stuck_count = 0

    for case in cases:
        result = run_case(case, r_pub, r_sub, args.timeout)
        all_results.append(result)
        if result["overall"] == "ok":
            ok_count += 1
        else:
            stuck_count += 1

        # Wait for any new nerves to finish all 3 qualification iterations
        _wait_for_qualification(r_pub)

        # Clear conversation context between cases to prevent bleed
        r_pub.delete("synapse:conversation")
        # Brief pause between cases
        time.sleep(2)

    # ── Summary ──
    print(f"\n\n{'='*60}")
    print(f"RESULTS: {ok_count} ok / {stuck_count} stuck / {len(cases)} total")
    print(f"{'='*60}")

    # Tag breakdown
    tag_stats = {}
    for r in all_results:
        t = r["tag"]
        if t not in tag_stats:
            tag_stats[t] = {"ok": 0, "stuck": 0}
        if r["overall"] == "ok":
            tag_stats[t]["ok"] += 1
        else:
            tag_stats[t]["stuck"] += 1

    print("\nBy tag:")
    for t, s in sorted(tag_stats.items()):
        total = s["ok"] + s["stuck"]
        print(f"  {t:20s}: {s['ok']}/{total} ok" + (f" ({s['stuck']} stuck)" if s["stuck"] else ""))

    if stuck_count:
        print("\nSTUCK cases:")
        for r in all_results:
            if r["overall"] == "STUCK":
                stuck_steps = [s for s in r["steps"] if s["status"] == "STUCK"]
                for s in stuck_steps:
                    print(f"  Case {r['id']} [{r['tag']}]: {s['msg'][:60]}")

    # ── Dreamstate wait ──
    if args.dream_wait > 0:
        print(f"\n{'='*60}")
        print(f"DREAMSTATE WAIT — pausing {args.dream_wait}s for brain to enter dreamstate")
        print(f"(Brain idle threshold is 120s — dreamstate starts after that)")
        print(f"{'='*60}")
        # Subscribe to brain:thought to see dreamstate events
        dream_ps = r_sub.pubsub()
        dream_ps.subscribe("brain:thought")
        dream_events = []
        start_wait = time.time()
        while time.time() - start_wait < args.dream_wait:
            msg = dream_ps.get_message(timeout=5)
            if msg and msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    stage = data.get("stage", "")
                    elapsed = int(time.time() - start_wait)
                    print(f"  [DREAM {elapsed}s] stage={stage}: {data.get('message', '')[:100]}", flush=True)
                    dream_events.append(data)
                except Exception:
                    pass
            else:
                elapsed = int(time.time() - start_wait)
                if elapsed % 30 == 0:
                    print(f"  [DREAM {elapsed}s] waiting...", flush=True)
        dream_ps.unsubscribe()
        dream_ps.close()
        print(f"\n  Captured {len(dream_events)} dreamstate events")

    # Save full report
    report_path = os.path.join(os.path.dirname(__file__), "stress_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved: {report_path}")


if __name__ == "__main__":
    main()
