# ElasticSearch Setup
Elasticsearch needs a higher `vm.max_map_count` (≥ 262144) for its memory-mapped files.

1. Edit the sysctl configuration file:

   ```bash
   sudo nano /etc/sysctl.conf
   ```

2. Add this line at the end (or edit if it already exists):

   ```
   vm.max_map_count=262144
   ```

3. Save and reload the settings without rebooting:

   ```bash
   sudo sysctl --system
   ```

4. Verify:

   ```bash
   sysctl vm.max_map_count
   # Should show: vm.max_map_count = 262144
   ```

Finally run the elastic container:
```
docker run --name es01 --net elastic -p 9200:9200 -d -m 1GB elasticsearch:9.1.2
```

reset and get the password for the user
```
docker exec -it es01 /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
```
