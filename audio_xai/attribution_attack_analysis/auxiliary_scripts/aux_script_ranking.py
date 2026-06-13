ast = [1, 3, 4, 2, 3, 5]
vgg = [2, 5, 7, 1, 4, 6]
spectra = [6, 8, 9, 7, 8, 9]

psycho = [1, 2, 8, 2, 4, 8]
pgd = [3, 5, 6, 3, 6, 7]
xsift = [4, 7, 9, 1, 5, 9]
import numpy as np

print(
    f"AST: {np.mean(ast):.3f} ± {np.std(ast, ddof=1) / np.sqrt(len(ast)):.3f}, median: {np.median(ast):.3f}"
)
print(
    f"VGG: {np.mean(vgg):.3f} ± {np.std(vgg, ddof=1) / np.sqrt(len(vgg)):.3f}, median: {np.median(vgg):.3f}"
)
print(
    f"SPECTRA: {np.mean(spectra):.3f} ± {np.std(spectra, ddof=1) / np.sqrt(len(spectra)):.3f}, median: {np.median(spectra):.3f}"
)
print(
    f"Psycho: {np.mean(psycho):.3f} ± {np.std(psycho, ddof=1) / np.sqrt(len(psycho)):.3f}, median: {np.median(psycho):.3f}"
)
print(
    f"PGD: {np.mean(pgd):.3f} ± {np.std(pgd, ddof=1) / np.sqrt(len(pgd)):.3f}, median: {np.median(pgd):.3f}"
)
print(
    f"XShift: {np.mean(xsift):.3f} ± {np.std(xsift, ddof=1) / np.sqrt(len(xsift)):.3f}, median: {np.median(xsift):.3f}"
)
