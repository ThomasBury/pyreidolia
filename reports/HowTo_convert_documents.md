# Using VSCode

 - Download the [markdown-pdf](https://marketplace.visualstudio.com/items?itemName=yzane.markdown-pdf) extension and [Chromium](https://download-chromium.appspot.com/) 
 - Open VSCode --> extension --> markdown-pdf --> extension setting --> scroll to markdown-pdf.executablePath  --> paste the `chromium.exe`  path 
 - Restart VSCode

It is all set up. For converting, `ctrl+shift+P` choose markdown to pdf or to HTML

# Using Pandoc:

**Disclaimer**:

pandoc is a powerful tool but uses its own mark flavour and uses LaTeX to convert to pdf. It is trickier than the previous method.

## DOCX to TEX

Convert a docx with embedded images to a markdown file.

```bash
pandoc --extract-media ./eda_files file.docx -o file.md
```

## MD to pdf

Using the `xelatex` engine and numbered TOC

```bash
pandoc --pdf-engine=xelatex --toc -N test.md -o test.pdf
```

## Other options

see [the pandoc demo page](https://pandoc.org/demos.html)

# Using NBCONVERT

Converting notebooks to ohter formats

## IPYNB to MD
```bash
jupyter nbconvert --output-dir=$report_path --no-input --to=markdown eda.ipynb
```

## IPYNB to LaTeX

```bash
jupyter nbconvert --output-dir=$report_path --no-input --to=latex eda.ipynb
```
