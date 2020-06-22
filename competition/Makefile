bundle.zip: \
    competition.yaml \
    overview.html \
    evaluation.html \
    terms_and_conditions.html \
    data.html \
    ingestion_program.zip \
    scoring_program.zip \
    public_data.zip \
    reference_data.zip \
    starting_kit.zip
	zip -j $@ $^

ingestion_program.zip: \
    ingestion_program/metadata \
    ingestion_program/ingestion.py
	zip -j $@ $^

scoring_program.zip: \
    scoring_program/metadata \
    scoring_program/scoring.py
	zip -j $@ $^

public_data.zip: \
    public_data/metadata \
    public_data/public_stock_data.txt
	zip -j $@ $^

reference_data.zip: \
    reference_data/metadata
	zip -j $@ $^

starting_kit.zip: \
    starting_kit/metadata
	zip -j $@ $^

clean:
	$(RM) *.zip

.PHONY: clean
