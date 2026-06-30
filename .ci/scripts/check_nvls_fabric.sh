#!/bin/bash -xe

echo "===== NVLS Fabric Smoke Test ($(hostname)) ====="

echo "INFO: Checking GPU driver ..."
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
NGPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPUS" -eq 0 ]; then
    echo "ERROR: No GPUs found"
    exit 1
fi
echo "INFO: Found $NGPUS GPUs"

echo "INFO: Checking NVLink fabric registration ..."
FABRIC_OUTPUT=$(nvidia-smi -q | grep 'Fabric' -A 4)
echo "$FABRIC_OUTPUT"

COMPLETED_COUNT=$(echo "$FABRIC_OUTPUT" | grep -c 'State.*:.*Completed' || true)
if [ "$COMPLETED_COUNT" -ne "$NGPUS" ]; then
    echo "ERROR: Expected $NGPUS GPUs with Fabric State 'Completed', found $COMPLETED_COUNT"
    exit 1
fi

FAILURES=$(echo "$FABRIC_OUTPUT" | grep 'Status' | grep -cv 'Success' || true)
if [ "$FAILURES" -ne 0 ]; then
    echo "ERROR: Some GPUs have Fabric Status != 'Success'"
    exit 1
fi
echo "INFO: All $NGPUS GPUs registered to NVLink fabric successfully"

echo "INFO: Checking NVLink link status ..."
nvidia-smi nvlink --status
echo "INFO: NVLink link status ... DONE"

echo "INFO: Checking GPU P2P topology ..."
nvidia-smi topo -p2p n
echo "INFO: GPU P2P topology ... DONE"

echo "===== NVLS Fabric Smoke Test PASSED ($(hostname)) ====="
