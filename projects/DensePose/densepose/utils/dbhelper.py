# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Dict, Optional, Tuple


class EntrySelector(object):
    """
    Base class for entry selectors
    """

    @staticmethod
    def from_string(spec: str) -> "EntrySelector":
        if spec == "*":
            return AllEntrySelector()
        return FieldEntrySelector(spec)


class AllEntrySelector(EntrySelector):
    """
    Selector that accepts all entries
    """

    SPECIFIER = "*"

    def __call__(self, entry):
        return True


class FieldEntrySelector(EntrySelector):
    """
    Selector that accepts only entries that match provided field
    specifier(s). Only a limited set of specifiers is supported for now:
      <specifiers>::=<specifier>[<comma><specifiers>]
      <specifier>::=<field_name>[<type_delim><type>]<equal><value_or_range>
      <field_name> is a valid identifier
      <type> ::= "int" | "str"
      <equal> ::= "="
      <comma> ::= ","
      <type_delim> ::= ":"
      <value_or_range> ::= <value> | <range>
      <range> ::= <value><range_delim><value>
      <range_delim> ::= "-"
      <value> is a string without spaces and special symbols
        (e.g. <comma>, <equal>, <type_delim>, <range_delim>)
    """

    _SPEC_DELIM = ","
    _TYPE_DELIM = ":"
    _RANGE_DELIM = "-"
    _EQUAL = "="
    _ERROR_PREFIX = "Invalid field selector specifier"

    class _FieldEntryValuePredicate(object):
        """
        Predicate that checks strict equality for the specified entry field
        """

        def __init__(self, name: str, typespec: Optional[str], value: str):
            import builtins

            self.name = name
            self.type = getattr(builtins, typespec) if typespec is not None else str
            self.value = value

        def __call__(self, entry):
            return entry[self.name] == self.type(self.value)

    class _FieldEntryRangePredicate(object):
        """
        Predicate that checks whether an entry field falls into the specified range
        """

        def __init__(self, name: str, typespec: Optional[str], vmin: str, vmax: str):
            import builtins

            self.name = name
            self.type = getattr(builtins, typespec) if typespec is not None else str
            self.vmin = vmin
            self.vmax = vmax

        def __call__(self, entry):
            return (entry[self.name] >= self.type(self.vmin)) and (
                entry[self.name] <= self.type(self.vmax)
            )

    def __init__(self, spec: str):
        self._predicates = self._parse_specifier_into_predicates(spec)

    def __call__(self, entry: Dict[str, Any]):
        for predicate in self._predicates:
            if not predicate(entry):
                return False
        return True

    def _parse_specifier_into_predicates(self, spec: str):
        predicates = []
        specs = spec.split(self._SPEC_DELIM)
        for subspec in specs:
            eq_idx = subspec.find(self._EQUAL)
            if eq_idx > 0:
                field_name_with_type = subspec[:eq_idx]
                field_name, field_type = self._parse_field_name_type(field_name_with_type)
                field_value_or_range = subspec[eq_idx + 1 :]
                if self._is_range_spec(field_value_or_range):
                    vmin, vmax = self._get_range_spec(field_value_or_range)
                    predicate = FieldEntrySelector._FieldEntryRangePredicate(
                        field_name, field_type, vmin, vmax
                    )
                else:
                    predicate = FieldEntrySelector._FieldEntryValuePredicate(
                        field_name, field_type, field_value_or_range
                    )
                predicates.append(predicate)
            elif eq_idx == 0:
                self._parse_error(f'"{subspec}", field name is empty!')
            else:
                self._parse_error(f'"{subspec}", should have format ' "<field>=<value_or_range>!")
        return predicates

    def _parse_field_name_type(self, field_name_with_type: str) -> Tuple[str, Optional[str]]:
        type_delim_idx = field_name_with_type.find(self._TYPE_DELIM)
        if type_delim_idx > 0:
            field_name = field_name_with_type[:type_delim_idx]
            field_type = field_name_with_type[type_delim_idx + 1 :]
        elif type_delim_idx == 0:
            self._parse_error(f'"{field_name_with_type}", field name is empty!')
        else:
            field_name = field_name_with_type
            field_type = None
        # pyre-fixme[61]: `field_name` may not be initialized here.
        # pyre-fixme[61]: `field_type` may not be initialized here.
        return field_name, field_type

    def _is_range_spec(self, field_value_or_range):
        delim_idx = field_value_or_range.find(self._RANGE_DELIM)
        return delim_idx > 0

    def _get_range_spec(self, field_value_or_range):
        if self._is_range_spec(field_value_or_range):
            delim_idx = field_value_or_range.find(self._RANGE_DELIM)
            vmin = field_value_or_range[:delim_idx]
            vmax = field_value_or_range[delim_idx + 1 :]
            return vmin, vmax
        else:
            self._parse_error('"field_value_or_range", range of values expected!')

    def _parse_error(self, msg):
        raise ValueError(f"{self._ERROR_PREFIX}: {msg}")
