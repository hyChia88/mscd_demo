======================================================================
  LoRA_4 PRE-TRAINING REVIEW
======================================================================

1. DATASET SIZE
   Train: 553
   Test:  75

2. SCHEMA VALIDATION
   Valid schema: 553/553

3. MESSAGE STRUCTURE
   Correct [system, user, assistant]: 553/553

4. IMAGE PATHS
   Images exist: 1021
   Images MISSING: 0

5. PREDICATE DISTRIBUTION
   ADJACENT_TO         :  182
   CONNECTS_TO         :  124
   CONTINUOUS          :   56
   FILLS               :  147
   By SR count: {0: 138, 1: 321, 2: 94}

6. PREDICATE VALUES
   All predicates valid ✓

7. IFC CLASS DISTRIBUTION
   IfcWallStandardCase      :  223
   IfcDoor                  :  154
   IfcWindow                :  131
   IfcSlab                  :   34
   IfcWall                  :    5
   IfcStair                 :    5
   IfcRailing               :    1

8. STOREY DISTRIBUTION
   1                        :  260
   -1                       :   92
   2                        :   65
   0                        :   61
   6                        :   22
   3                        :   19
   4                        :   15
   5                        :    9
   T/FDN                    :    6
   roof                     :    4

9. CONFIDENCE VALUES
   confidence=0.8: 94
   confidence=1.0: 415

10. SYSTEM PROMPT
   "You are a construction site assistant that extract...": 553

11. FLOORPLAN-SR COUPLING
   ✓ No SR-without-floorplan records

12. DUPLICATE CHECK
   Unique IDs: 553/553

======================================================================
  REVIEW COMPLETE
======================================================================