# LLMpedia Visual Design Guide

## Design Philosophy

LLMpedia follows a **sophisticated academic minimalism** that combines scholarly authority with digital precision. The visual identity honors arXiv's research heritage while embracing contemporary web design principles and systematic design thinking.

### Core Design Principles

#### **1. Academic Minimalism**
- Clean typography with generous whitespace for enhanced readability
- Purposeful use of color and visual elements - every element serves a function
- Professional appearance suitable for research content consumption
- Focus on content hierarchy and information architecture

#### **2. arXiv-Inspired Color Palette**
- **Primary Brand**: arXiv red (`#b31b1b`) with variations (`#c93232` light, `#8f1414` dark)  
- **Purpose**: Maintains visual connection to the academic ecosystem
- **Application**: Interactive elements, accents, gradients, call-to-action components
- **Restraint**: Used strategically as accent color, not overwhelming

#### **3. Zen-Like Interaction Design**
- Subtle hover effects and smooth micro-interactions
- Gentle elevation and shadow systems that enhance without distraction
- Consistent transition timing across all components
- Touch-friendly mobile interactions with 44px minimum tap targets

#### **4. Sophisticated Academic Typography**
- **Libertinus Serif typeface** for headers - scholarly authority with contemporary clarity
- Clean borders and consistent border radius scale  
- Structured layouts with precise spacing relationships
- Digital precision through geometric elements and systematic design tokens

## Design System Architecture

### **CSS Custom Properties (Design Tokens)**
All visual properties are centralized using CSS custom properties for consistency:

```css
:root {
  /* Brand Colors */
  --arxiv-red: #b31b1b;
  --arxiv-red-light: #c93232;
  --arxiv-red-dark: #8f1414;
  
  /* Surface Gradients */
  --surface-light: #ffffff;
  --surface-light-alt: #fafbfc;
  --surface-dark: #1e1e1e;
  --surface-dark-alt: #252526;
  
  /* Typography Scale */
  --font-family-display: 'Libertinus Serif', 'Latin Modern Roman', 'Computer Modern', serif;
  --font-family-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --font-family-mono: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  
  /* Spacing Scale */
  --space-xs: 0.25rem;
  --space-sm: 0.5rem;
  --space-base: 1rem;
  --space-lg: 1.5rem;
  --space-xl: 2rem;
  --space-2xl: 3rem;
}
```

### **Dark Mode Support**
- Automatic theme detection using `prefers-color-scheme`
- Enhanced contrast ratios for accessibility  
- Subtle shadow adjustments for dark environments
- Consistent color relationships across themes

## Available Components

### **1. Base Component Library**

#### **Card System**
- **`.card-base`**: Foundation card with hover effects and gradient backgrounds
- **Hover States**: Subtle lift animation with enhanced shadows
- **Accent Bars**: Top gradient bars that appear on hover
- **Responsive**: Adapts padding and margins on mobile

#### **Button Hierarchy**
- **`.btn-primary`**: Gradient background, white text, hover lift
- **`.btn-secondary`**: Transparent with red border, fills on hover  
- **Streamlit Overrides**: Custom styling for all Streamlit button types
- **Tertiary Buttons**: Subtle border, background fill on hover

#### **Typography System**
- **`.heading-display`**: Large display headers with Libertinus Serif font
- **`.heading-geometric`**: Medium headers with academic spacing
- **`.text-monospace`**: Code and technical content
- **`.text-precise`**: Optimized line-height for body text

### **2. Layout Utilities**

#### **Grid System**
- **`.content-grid`**: 2fr + 1px + 1fr layout with separator line
- **`.content-grid-separator`**: Vertical gradient separator
- **`.grid-auto-fit`**: Responsive auto-fitting grid

#### **Flexbox Utilities**
- **`.centered`**: Horizontally centered flex container
- **`.flex-between`**: Space-between with center alignment
- **`.flex-center`**: Full center alignment

### **3. Specialized Components**

#### **Trending Card System**
- **`.trending-card`**: Complex card with image, content, and metadata
- **`.trending-image`**: 90px square with hover scale and rotation
- **`.trending-punchline`**: Descriptive text with optimal readability
- **`.trending-metadata`**: Author and metrics layout
- **Badge System**: Category and metric badges with hover animations

#### **Tweet Timeline Components**  
- **`.tweet-timeline-container`**: Carousel container with horizontal scroll
- **`.tweet-card`**: Fixed-width cards (320px) with vertical content
- **`.tweet-carousel`**: Horizontal scrolling with custom scrollbars
- **Custom Scrollbars**: Themed scrollbars matching brand colors

#### **Featured Card**
- **`.featured-card`**: Hero card with large image (280px height)
- **`.featured-image`**: Cover image with subtle scale on hover
- **`.featured-content`**: Structured content area with title and description
- **Text Clamping**: 6-line limit with ellipsis for descriptions

#### **Interesting Facts Cards**
- **`.fact-card`**: Left border accent cards
- **`.fact-metadata`**: Topic badges and paper links
- **Hover Effects**: Border color transitions and elevation

#### **Research Portal**
- **`.research-portal`**: Main search interface with gradient background
- **`.suggestion-chip`**: Interactive suggestion buttons
- **`.research-suggestions`**: Flexible chip container

### **4. Header System**

#### **Trending Panel Headers**
- **`.trending-panel-header`**: Consistent section headers across the app
- **`.trending-panel-title`**: Large title with Orbitron font
- **`.trending-panel-subtitle`**: Descriptive subtitle text
- **Top Gradient Bar**: Subtle brand color accent

### **5. Interactive Elements**

#### **Discovery Actions**
- **`.discovery-actions`**: Hover-revealed action buttons
- **`.action-btn`**: Small interactive buttons with hover states
- **`.preview-tooltip`**: Contextual information on hover

#### **Geometric Dividers**
- **`.geometric-divider`**: Horizontal gradient dividers
- **`.main-section`**: Section containers with automatic dividers

### **6. Mobile Responsive System**

#### **Mobile-First Approach**
- **Single Mobile Breakpoint**: 768px and below
- **Touch-Optimized**: 44px minimum touch targets
- **Stacked Layouts**: Grid systems become single column
- **Readable Typography**: Larger font sizes on mobile

#### **Component Adaptations**
- **Cards**: Reduced padding and margins
- **Images**: Smaller dimensions (60px vs 90px for trending images)
- **Navigation**: Column-based layouts become stacked
- **Buttons**: Full-width with increased padding

## Usage Guidelines

### **1. Color Usage**
- **Primary Red**: Use sparingly for key actions and accents
- **Gradients**: Subtle surface gradients, avoid heavy color overlays
- **Text Colors**: Maintain sufficient contrast ratios (4.5:1 minimum)

### **2. Typography Hierarchy**
- **Display Headers**: Use Orbitron for main page titles
- **Section Headers**: Geometric headers for content sections  
- **Body Text**: System fonts for optimal readability
- **Code/Data**: Monospace fonts for technical content

### **3. Animation Timing**
- **Fast**: 0.15s for micro-interactions (hover states)
- **Base**: 0.3s for standard transitions (card animations)
- **Slow**: 0.6s for complex animations (flip cards)

### **4. Component Composition**
- **Consistent Spacing**: Use design token spacing scale
- **Nested Components**: Cards can contain other components
- **Responsive Behavior**: All components adapt to mobile automatically

## Implementation Notes

### **CSS Organization**
The styling system (`utils/styling.py`) is organized into logical sections:
1. **Design Tokens**: CSS custom properties
2. **Base Components**: Reusable patterns
3. **Layout Utilities**: Grid and flex systems  
4. **Specialized Components**: Feature-specific styles
5. **Streamlit Overrides**: Framework customizations
6. **Mobile Responsive**: Consolidated mobile styles

### **Performance Considerations**
- **CSS Custom Properties**: Efficient browser rendering
- **Minimal Reflows**: Careful use of transforms vs layout properties
- **Lightweight Animations**: GPU-accelerated transforms
- **Efficient Selectors**: Avoid deep nesting and universal selectors

### **Maintenance Guidelines**
- **Single Source of Truth**: All styles managed in `utils/styling.py`
- **Component Isolation**: Styles scoped to component functions
- **No Inline Styles**: All styling through CSS classes
- **Design Token Usage**: Always use CSS custom properties for consistency

---

*This visual guide reflects the current state of LLMpedia's design system. For implementation details, refer to `utils/styling.py`. For project structure, see `STRUCTURE.md`.*