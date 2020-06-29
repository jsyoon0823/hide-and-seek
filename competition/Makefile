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
	$(RM) $@
	zip $@ $^

ingestion_program.zip: \
    ingestion_program/metadata \
    ingestion_program/ingestion.py \
    ingestion_program/data/data_preprocess.py \
    ingestion_program/data/data_utils.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

scoring_program.zip: \
    scoring_program/metadata \
    scoring_program/scoring.py \
    scoring_program/metrics/general_rnn.py \
    scoring_program/metrics/metric_utils.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

public_data.zip: \
    public_data/metadata \
    public_data/public_stock_data.txt
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

reference_data.zip: \
    reference_data/metadata
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

starting_kit/hider_add_noise.zip: \
    starting_kit/hider_add_noise/metadata \
    starting_kit/hider_add_noise/hider.py \
    starting_kit/hider_add_noise/add_noise.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

starting_kit/hider_timegan.zip: \
    starting_kit/hider_timegan/metadata \
    starting_kit/hider_timegan/hider.py \
    starting_kit/hider_timegan/timegan/timegan.py \
    starting_kit/hider_timegan/timegan/utils.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

starting_kit/seeker_binary_predictor.zip: \
    starting_kit/seeker_binary_predictor/metadata \
    starting_kit/seeker_binary_predictor/seeker.py \
    starting_kit/seeker_binary_predictor/binary_predictor/binary_predictor.py \
    starting_kit/seeker_binary_predictor/binary_predictor/general_rnn.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

starting_kit/seeker_knn.zip: \
    starting_kit/seeker_knn/metadata \
    starting_kit/seeker_knn/seeker.py \
    starting_kit/seeker_knn/knn_seeker.py
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

starting_kit.zip: \
    starting_kit/hider_add_noise.zip \
    starting_kit/hider_timegan.zip \
    starting_kit/seeker_binary_predictor.zip \
    starting_kit/seeker_knn.zip
	$(RM) $@
	cd $(@:%.zip=%) && zip $(abspath $@) $(^:$(@:%.zip=%)/%=%)

clean:
	$(RM) *.zip starting_kit/*.zip

.PHONY: clean
