import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

entries = np.array(
    [
        ## elders
        [22, 200, 3],
        [71, 200, 3],
        [70, 200, 3],
        [69, 200, 3],
        [80, 200, 3],
        [16, 200, 3],
        [46, 200, 3],
        [34, 200, 3],
        ## quick
        [61, 200, 3],
        [61, 100, 3],
        [61, 20, 3],
        [61, 10, 3],
        [61, 15, 3],
        [61, 16, 3],
        [61, 6, 3],
        [61, 1, 3],
        ## relationship
        [61, 200, 4],
        [61, 200, 5],
        [21, 100, 4],
        [21, 100, 5],
        [21, 100, 3],
        ## conflicts
        [61, 2, 5],
        [71, 2, 5],
        [71, 2, 4],
        [71, 100, 4],
    ],
    dtype=float,
)

labels = {"std": "standard", "hp": "high priority", "qck": "quick"}


entries_labels = np.array(
    [
        ## elders
        labels["std"],
        labels["hp"],
        labels["hp"],
        labels["std"],
        labels["hp"],
        labels["std"],
        labels["std"],
        labels["std"],
        ## quick
        labels["std"],
        labels["std"],
        labels["std"],
        labels["qck"],
        labels["qck"],
        labels["std"],
        labels["qck"],
        labels["qck"],
        ## relationship
        labels["hp"],
        labels["hp"],
        labels["hp"],
        labels["hp"],
        labels["std"],
        ##conflicts
        labels["qck"],
        labels["qck"],
        labels["qck"],
        labels["hp"],
    ]
)

scaler = StandardScaler()
normalized = scaler.fit_transform(entries)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(normalized, entries_labels)

print(model.predict(scaler.transform(np.array([[22, 10, 2]]))))
