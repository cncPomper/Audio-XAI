## AASR — Attribution Attack Success Rate

**AASR** określa odsetek próbek, dla których atak na wyjaśnienie modelu XAI zakończył się sukcesem, przy jednoczesnym zachowaniu decyzji modelu oraz jakości próbki adversarialnej.

Atak uznajemy za skuteczny wtedy, gdy spełnione są jednocześnie następujące warunki:

1. model zachowuje tę samą predykcję dla próbki oryginalnej i adversarialnej,
2. podobieństwo kosinusowe między atrybucją oryginalną i adversarialną spada poniżej ustalonego progu,
3. pokrycie najważniejszych cech `Top-k` spada poniżej ustalonego progu,
4. jakość próbki adversarialnej pozostaje powyżej minimalnego wymaganego poziomu.

Formalnie:

$$
AASR = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}
\left[
P_i \land C_i \land T_i \land Q_i
\right]
$$

gdzie:

$$
P_i =
\mathbb{1}
\left[
\hat{y}^{orig}_i = \hat{y}^{adv}_i
\right]
$$

oznacza zachowanie tej samej predykcji przez model,

$$
C_i =
\mathbb{1}
\left[
\cos(A^{orig}_i, A^{adv}_i) < \tau_{cos}
\right]
$$

oznacza wystarczająco dużą zmianę globalnej struktury mapy atrybucji,

$$
T_i =
\mathbb{1}
\left[
TopKOverlap(A^{orig}_i, A^{adv}_i) < \tau_{topk}
\right]
$$

oznacza zmianę zbioru najważniejszych cech według wyjaśnienia XAI,

oraz:

$$
Q_i =
\mathbb{1}
\left[
Quality_i \geq \tau_{quality}
\right]
$$

oznacza zachowanie minimalnej jakości próbki adversarialnej.

Ostatecznie dla pojedynczej próbki:

$$
aasr_i =
\mathbb{1}
\left[
\hat{y}^{orig}_i = \hat{y}^{adv}_i
\land
\cos(A^{orig}_i, A^{adv}_i) < \tau_{cos}
\land
TopKOverlap(A^{orig}_i, A^{adv}_i) < \tau_{topk}
\land
Quality_i \geq \tau_{quality}
\right]
$$

a dla całego zbioru:

$$
AASR = \frac{\sum_{i=1}^{N} aasr_i}{N}
$$

### Przykładowe obliczanie

Załóżmy, że analizujemy `N = 5` próbek oraz stosujemy progi:

$$
\tau_{cos} = 0.20
$$

$$
\tau_{topk} = 0.10
$$

$$
\tau_{quality} = 0.80
$$

| Próbka | Predykcja zachowana | Cosine similarity | Top-k overlap | Quality | Sukces AASR |
|---:|:---:|---:|---:|---:|:---:|
| 1 | Tak | 0.12 | 0.05 | 0.91 | 1 |
| 2 | Tak | 0.31 | 0.04 | 0.88 | 0 |
| 3 | Nie | 0.10 | 0.03 | 0.94 | 0 |
| 4 | Tak | 0.18 | 0.08 | 0.76 | 0 |
| 5 | Tak | 0.07 | 0.02 | 0.86 | 1 |

Sukces uzyskały próbki `1` oraz `5`, więc:

$$
AASR = \frac{2}{5} = 0.40
$$

czyli:

$$
AASR = 40\%
$$

---

## AFS stable — Attribution Fragility Score Stable

**AFS stable** mierzy siłę zmiany wyjaśnienia XAI po ataku, ale premiuje tylko takie przypadki, w których model zachował pierwotną decyzję oraz jakość próbki adversarialnej pozostała akceptowalna.

Metryka ta jest bardziej informatywna niż samo AASR, ponieważ nie mówi jedynie, czy atak przekroczył ustalone progi, ale określa ciągły poziom podatności wyjaśnienia na zaburzenie.

AFS stable można interpretować jako:

> jak mocno zmieniło się wyjaśnienie modelu, przy zachowaniu tej samej decyzji modelu i wystarczającej jakości próbki adversarialnej.

Najpierw definiujemy zmianę atrybucji:

$$
AttributionChange_i =
1 -
\frac{
\cos(A^{orig}_i, A^{adv}_i) + TopKOverlap(A^{orig}_i, A^{adv}_i)
}{2}
$$

gdzie:

- `cos(A_orig, A_adv)` mierzy podobieństwo globalnej struktury map atrybucji,
- `TopKOverlap(A_orig, A_adv)` mierzy podobieństwo zbioru najbardziej istotnych cech,
- im niższe podobieństwo, tym większa zmiana wyjaśnienia.

Następnie uwzględniamy stabilność decyzji modelu:

$$
PredictionPreserved_i =
\mathbb{1}
\left[
\hat{y}^{orig}_i = \hat{y}^{adv}_i
\right]
$$

oraz jakość próbki adversarialnej:

$$
QualityScore_i \in [0, 1]
$$

Wtedy:

$$
AFS^{stable}_i =
AttributionChange_i
\cdot
PredictionPreserved_i
\cdot
QualityScore_i
$$

czyli po podstawieniu:

$$
AFS^{stable}_i =
\left(
1 -
\frac{
\cos(A^{orig}_i, A^{adv}_i) + TopKOverlap(A^{orig}_i, A^{adv}_i)
}{2}
\right)
\cdot
\mathbb{1}
\left[
\hat{y}^{orig}_i = \hat{y}^{adv}_i
\right]
\cdot
QualityScore_i
$$

Dla całego zbioru danych:

$$
AFS^{stable} =
\frac{1}{N}
\sum_{i=1}^{N}
AFS^{stable}_i
$$

### Interpretacja wartości

| Wartość AFS stable | Interpretacja |
|---:|---|
| blisko `0.0` | atak nie zmienił znacząco wyjaśnienia albo zmienił predykcję modelu |
| około `0.5` | umiarkowana zmiana wyjaśnienia przy częściowym zachowaniu jakości |
| blisko `1.0` | bardzo silna zmiana wyjaśnienia, predykcja zachowana, jakość wysoka |

### Przykładowe obliczanie

Załóżmy, że dla jednej próbki mamy:

$$
\cos(A^{orig}, A^{adv}) = 0.20
$$

$$
TopKOverlap(A^{orig}, A^{adv}) = 0.10
$$

$$
PredictionPreserved = 1
$$

$$
QualityScore = 0.90
$$

Najpierw liczymy zmianę atrybucji:

$$
AttributionChange =
1 -
\frac{0.20 + 0.10}{2}
$$

$$
AttributionChange =
1 - 0.15
$$

$$
AttributionChange = 0.85
$$

Następnie:

$$
AFS^{stable} =
0.85 \cdot 1 \cdot 0.90
$$

$$
AFS^{stable} = 0.765
$$

Oznacza to, że wyjaśnienie modelu zostało silnie zmienione, przy zachowaniu decyzji modelu i dobrej jakości próbki adversarialnej.

---

## Różnica między AASR a AFS stable

| Cecha | AASR | AFS stable |
|---|---|---|
| Typ metryki | binarna / progowa | ciągła |
| Wynik pojedynczej próbki | `0` albo `1` | wartość z zakresu `[0, 1]` |
| Zależy od progów | tak | nie bezpośrednio |
| Uwzględnia zachowanie predykcji | tak | tak |
| Uwzględnia jakość próbki | tak | tak |
| Mierzy siłę zmiany wyjaśnienia | pośrednio | bezpośrednio |
| Dobra do raportowania skuteczności ataku | tak | tak |
| Dobra do rankingu podatności wyjaśnień | częściowo | bardzo dobrze |

W praktyce warto raportować obie metryki:

- **AASR** pokazuje, jak często atak spełnia ścisłe kryteria sukcesu.
- **AFS stable** pokazuje, jak silnie atak zmienia wyjaśnienia przy zachowaniu decyzji modelu i jakości próbki.