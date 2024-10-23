run: docker-build
	docker run -p 8090:80 mclaps

docker-build:
	docker build -t mclaps .