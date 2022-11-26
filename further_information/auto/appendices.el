(TeX-add-style-hook
 "appendices"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10")
   (LaTeX-add-labels
    "tab:artifical_vocabulary"
    "tab:hyperparameter_space"))
 :latex)

