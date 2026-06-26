import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────
np.random.seed(42)

hours = np.array([
    1.0, 1.5, 2.0, 2.2, 2.5, 3.0, 3.5, 3.8,
    4.0, 4.2, 4.5, 5.0, 5.2, 5.5, 6.0, 6.5,
    6.8, 7.0, 7.2, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0
])
scores = hours * 9 + 22 + np.random.normal(0, 2, len(hours))
scores = np.clip(scores, 0, 100)

df = pd.DataFrame({'hours_studied': hours, 'score': scores})

print("=" * 50)
print("        STUDENT SCORE PREDICTOR")
print("=" * 50)
print(f"\nTotal dataset size : {len(df)} students")
print("\nSample data (first 5 rows):")
print(df.head().to_string(index=False))

# ─────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X = df[['hours_studied']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ─────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\nModel equation   : score = {model.coef_[0]:.4f} × hours + {model.intercept_:.4f}")

# ─────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("  EVALUATION RESULTS")
print("=" * 50)
print(f"  Mean Absolute Error (MAE) : {mae:.2f} points")
print(f"  R² Score                  : {r2:.4f}")
print("=" * 50)

# ─────────────────────────────────────────
#  PREDICTIONS vs ACTUALS TABLE
# ─────────────────────────────────────────
results = pd.DataFrame({
    'Hours'    : X_test['hours_studied'].values,
    'Actual'   : y_test.values.round(2),
    'Predicted': y_pred.round(2),
    'Error'    : (y_pred - y_test.values).round(2)
}).reset_index(drop=True)
results.index += 1
results.index.name = 'Student'

print("\nPredictions vs Actuals:")
print(results.to_string())

# ─────────────────────────────────────────
#  PREDICT FOR A NEW STUDENT
# ─────────────────────────────────────────
new_hours = 4.5
predicted_score = model.predict([[new_hours]])[0]
print(f"\nPredicted score for {new_hours} hours of study : {predicted_score:.2f}")

# ─────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Student Score Predictor — Model Evaluation", fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Colour palette
C_BLUE  = '#2a78d6'
C_GREEN = '#1baf7a'
C_GRAY  = '#888780'
C_RED   = '#e34948'

# ── Plot 1: Regression line ──────────────
ax1 = fig.add_subplot(gs[0, :])
h_line = np.linspace(hours.min(), hours.max(), 200).reshape(-1, 1)
ax1.scatter(X_train, y_train, color=C_GRAY,  alpha=0.7, s=60, label='Training data', zorder=3)
ax1.scatter(X_test,  y_test,  color=C_GREEN, alpha=0.9, s=80, label='Test data',     zorder=4)
ax1.plot(h_line, model.predict(h_line), color=C_BLUE, linewidth=2.5, label='Regression line', zorder=2)
ax1.scatter([new_hours], [predicted_score], color='orange', s=130, zorder=5,
            marker='*', label=f'New student ({new_hours}h → {predicted_score:.1f})')
ax1.set_xlabel('Hours Studied', fontsize=11)
ax1.set_ylabel('Exam Score', fontsize=11)
ax1.set_title('Hours Studied vs Exam Score', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_ylim(0, 115)

# ── Plot 2: Actual vs Predicted bar chart ──
ax2 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(len(y_test))
width = 0.35
ax2.bar(x_pos - width/2, y_test.values,  width, color=C_GREEN, alpha=0.85, label='Actual')
ax2.bar(x_pos + width/2, y_pred,          width, color=C_BLUE,  alpha=0.85, label='Predicted')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'S{i+1}' for i in range(len(y_test))], fontsize=10)
ax2.set_xlabel('Test Student', fontsize=11)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Actual vs Predicted Scores', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', alpha=0.25)
ax2.set_ylim(0, 115)

# ── Plot 3: Scatter actual vs predicted ──
ax3 = fig.add_subplot(gs[1, 1])
min_v = min(y_test.min(), y_pred.min()) - 5
max_v = max(y_test.max(), y_pred.max()) + 5
ax3.scatter(y_test, y_pred, color=C_BLUE, s=80, alpha=0.85, zorder=3)
ax3.plot([min_v, max_v], [min_v, max_v], color=C_GRAY, linewidth=1.5,
         linestyle='--', label='Perfect prediction', zorder=2)
for actual, pred in zip(y_test.values, y_pred):
    ax3.plot([actual, actual], [actual, pred],
             color=C_RED, linewidth=1, alpha=0.5, zorder=1)
ax3.set_xlabel('Actual Score', fontsize=11)
ax3.set_ylabel('Predicted Score', fontsize=11)
ax3.set_title('Actual vs Predicted (Scatter)', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.25)
ax3.text(0.05, 0.92, f'MAE = {mae:.2f} pts\nR²  = {r2:.3f}',
         transform=ax3.transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.savefig('student_score_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'student_score_results.png'")
plt.show()
