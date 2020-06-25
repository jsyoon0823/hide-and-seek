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

starting_kit/hider_add_noise.zip: \
    starting_kit/hider_add_noise/metadata \
    starting_kit/hider_add_noise/add_noise.py
	zip -j $@ $^

starting_kit/hider_timegan.zip: \
    starting_kit/hider_timegan/metadata \
    starting_kit/hider_timegan/timegan.py \
    starting_kit/hider_timegan/utils.py
	zip -j $@ $^

starting_kit/seeker_binary_predictor.zip: \
    starting_kit/seeker_binary_predictor/metadata \
    starting_kit/seeker_binary_predictor/binary_predictor.py \
    starting_kit/seeker_binary_predictor/general_rnn.py
	zip -j $@ $^

starting_kit/seeker_knn.zip: \
    starting_kit/seeker_knn/metadata \
    starting_kit/seeker_knn/knn_seeker.py
	zip -j $@ $^

starting_kit.zip: \
    starting_kit/hider_add_noise.zip \
    starting_kit/hider_timegan.zip \
    starting_kit/seeker_binary_predictor.zip \
    starting_kit/seeker_knn.zip
	zip -j $@ $^

clean:
	$(RM) *.zip starting_kit/*.zip

.PHONY: clean
