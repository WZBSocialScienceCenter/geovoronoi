sdist:
	python setup.py sdist

wheel:
	python setup.py bdist_wheel

pypi_upload:
	python setup.py sdist bdist_wheel upload


