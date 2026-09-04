<script setup>
const props = defineProps({
  spec: { type: Object, required: true },
  modelValue: { required: true },
});

const emit = defineEmits(["update:modelValue", "change"]);

function onInput(event) {
  const spec = props.spec;
  let value;
  if (spec.type === "bool") value = event.target.checked;
  else if (spec.type === "int") value = event.target.value === "" ? null : Number(event.target.value);
  else if (spec.type === "float") value = event.target.value === "" ? null : Number(event.target.value);
  else value = event.target.value;
  emit("update:modelValue", value);
  emit("change");
}
</script>

<template>
  <div class="field">
    <label>{{ spec.label }}</label>
    <select
      v-if="spec.type === 'enum'"
      :value="modelValue"
      @change="onInput"
    >
      <option v-for="choice in spec.choices || []" :key="choice" :value="choice">
        {{ choice }}
      </option>
    </select>
    <input
      v-else-if="spec.type === 'bool'"
      type="checkbox"
      :checked="!!modelValue"
      @change="onInput"
    />
    <input
      v-else
      type="number"
      :step="spec.type === 'int' ? '1' : 'any'"
      :min="spec.min"
      :max="spec.max"
      :value="modelValue"
      @change="onInput"
      @input="onInput"
    />
  </div>
</template>
