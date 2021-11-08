if [ -d "benign_train" -a -d "benign_test" -a -d "spectre_train" -a -d "spectre_test" ]; then
    echo spectre data already exist
else
    cat spectre_data* > spectre_data.tar.gz
	tar -xzvf spectre_data.tar.gz
	rm spectre_data.tar.gz
	echo spectre data extraction completed
fi
