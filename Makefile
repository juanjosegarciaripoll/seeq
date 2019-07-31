CLEANUP=jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace

all:
	python -c 'import exportnb; import glob; exportnb.export_notebooks(glob.glob("*.ipynb")); quit()' 

clean:
	rm -rf seeq

cleanup:
	for i in *.ipynb; do \
	  $(CLEANUP) "$$i"; \
	done
