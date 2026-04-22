# Frontend Enhancements - Deepfake Detector

## Overview
Enhanced the React frontend with advanced UI components including data visualizations, image comparison slider, and improved metrics display.

## New Features Implemented

### 1. **Confidence Meter (Progress Bar)**
- **Location**: Analysis Tab > Result Section
- **Features**:
  - Visual progress bar showing detection confidence (0-100%)
  - Color-coded (red for FAKE, green for REAL)
  - Percentage label displayed next to meter
  - Confidence markers at 0%, 50%, and 100%
  - Smooth animation when confidence is calculated
- **Styling**: Gradient fills, responsive design

### 2. **Heatmap Overlay Toggle**
- **Location**: Image Analysis Section
- **Features**:
  - Two-mode toggle: "Overlay" and "Compare"
  - **Overlay Mode**: Shows GradCAM heatmap overlaid on original image with 70% opacity
  - **Compare Mode**: Activates interactive comparison slider
  - Active button highlighting to indicate current mode

### 3. **Before vs After Comparison Slider**
- **Location**: Image Analysis Section (Compare mode)
- **Features**:
  - Interactive slider to compare original image with AI explanation heatmap
  - Mouse-move tracking for smooth slider control
  - Draggable handle with visual indicator
  - Left/right labels ("Original" / "AI Explanation")
  - Smooth clipping animation during slider movement
  - Desktop-optimized with cursor feedback

### 4. **Robustness Test Results Visualization**
- **Location**: Robustness Tab
- **Features**:
  - **Bar Chart**: Displays confidence scores and inference times for each perturbation
  - Shows all 7 perturbation types:
    - Original
    - Gaussian Noise
    - JPEG Compression
    - Brightness/Contrast
    - Blur
    - Rotation
    - Scaling
  - Dual-axis chart (confidence on left, time on right)
  - Color-coded bars (#6366f1 for confidence, #ec4899 for time)
  - Interactive tooltips on hover

### 5. **Comprehensive Metrics Charts**
- **Location**: Metrics Tab
- **Features**:

#### a) **Overall Performance Metrics Chart**
- Bar chart showing key metrics:
  - Accuracy
  - Precision (Macro & Weighted)
  - Recall (Macro & Weighted)
  - F1 Score (Macro & Weighted)
  - AUC-ROC
- Y-axis range: 0-100%
- Color: #6366f1 (Indigo)

#### b) **Per-Class Performance Chart**
- Grouped bar chart for Fake vs Real classes
- Metrics displayed:
  - Precision
  - Recall
  - F1 Score
- Different colors for each metric
- Direct comparison between classes

#### c) **Confusion Matrix Pie Chart**
- Visual breakdown of prediction outcomes:
  - True Negative (Indigo)
  - False Positive (Pink)
  - False Negative (Orange)
  - True Positive (Green)
- Shows actual count values
- Interactive legend

## Technical Implementation

### Dependencies Added
```bash
npm install recharts
```

### New React Components

#### 1. **ComparisonSlider**
```jsx
function ComparisonSlider({ originalImage, heatmapImage })
```
- Handles mouse movement tracking
- Manages slider position state (0-100%)
- Clips heatmap image based on slider position

#### 2. **RobustnessChart**
```jsx
function RobustnessChart({ results })
```
- Transforms perturbation results into chart data
- Renders dual-axis BarChart with confidence and time metrics
- Includes cartesian grid, axes, and tooltip

#### 3. **MetricsChart**
```jsx
function MetricsChart({ metrics })
```
- Generates three separate visualizations
- Handles data transformation from API response
- Uses BarChart and PieChart components
- Color-coded elements for easy interpretation

### State Management
Added new state variable for comparison slider:
```jsx
const [useComparisonSlider, setUseComparisonSlider] = useState(false);
```

### API Integration
- Backend URL: `http://127.0.0.1:5000`
- Endpoints used:
  - `/predict` - Image analysis with heatmap
  - `/robustness-test` - Robustness test results
  - `/metrics` - Model metrics

## Styling Enhancements

### CSS Classes Added
- `.comparison-slider` - Container for comparison component
- `.comparison-container` - Layout for original and heatmap images
- `.comparison-slider-handle` - Draggable handle with arrows
- `.slider-arrow` - Visual indicator on slider handle
- `.heatmap-controls` - Toggle buttons for overlay/compare modes
- `.confidence-section` - Enhanced confidence meter display
- `.confidence-markers` - Percentage markers (0%, 50%, 100%)
- `.result-stats` - Badges for inference time and model accuracy
- `.stat-badge` - Individual stat badge styling
- `.chart-container` - Wrapper for chart components
- `.metrics-charts` - Grid layout for multiple charts

### Design Principles
- **Color Scheme**: Consistent with existing indigo (#6366f1) primary color
- **Responsive**: Mobile-optimized with media queries
- **Accessibility**: Clear labels, high contrast, readable fonts
- **Performance**: Optimized chart rendering with RecHarts

## User Experience Flow

### Analysis Tab
1. User uploads image
2. Clicks "Analyze Image"
3. Results displayed with:
   - Confidence meter (visual progress bar)
   - Detection verdict (FAKE/REAL)
   - Inference timing
4. User can toggle between:
   - **Overlay Mode**: See heatmap overlaid on image
   - **Compare Mode**: Interactive slider to compare original vs heatmap

### Robustness Tab
1. User runs robustness test
2. Views summary statistics
3. **Chart visualization** shows:
   - How confidence varies across perturbations
   - Inference time for each transformation
4. Detailed grid below shows individual results

### Metrics Tab
1. User fetches model metrics
2. **Three charts displayed**:
   - Overall performance metrics (bar chart)
   - Per-class comparison (grouped bar chart)
   - Confusion matrix breakdown (pie chart)
3. Additional information tables below charts

## Browser Compatibility
- Modern browsers with ES6+ support
- Chrome, Firefox, Safari, Edge
- Responsive design tested at 320px to 1920px widths

## Future Enhancement Opportunities
1. Export chart data as CSV
2. Zoom and pan functionality for images
3. Multiple image batch upload
4. Custom perturbation parameters
5. Real-time chart updates
6. Dark mode support
7. Animation transitions for chart rendering
8. Touchscreen support for comparison slider

## Performance Notes
- Charts use ResponsiveContainer for automatic sizing
- Lazy loading can be implemented for large datasets
- Memoization recommended for complex chart data
- SVG rendering optimized by RecHarts

## Testing Checklist
- [ ] Upload image and verify confidence meter displays correctly
- [ ] Test overlay toggle functionality
- [ ] Test comparison slider mouse tracking
- [ ] Run robustness test and verify bar chart renders
- [ ] Check robustness test results grid below chart
- [ ] Fetch metrics and verify all three charts render
- [ ] Test responsive design on mobile (portrait/landscape)
- [ ] Verify API calls to correct port (5000)
- [ ] Check chart colors and labels are correct
- [ ] Test on different browsers

## Installation & Running

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

Server runs on: `http://localhost:5175` (or next available port)
Backend API: `http://127.0.0.1:5000`
