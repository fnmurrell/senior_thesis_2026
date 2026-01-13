.PHONY: run_pipeline
run_pipeline: 
	python main.py

.PHONY: clean
clean:
	rm goodreads_reviews.json
	rm goodreads_eng_only_reviews.json